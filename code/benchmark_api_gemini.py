import argparse
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from benchmark_common import build_history, init_output_payload
from common import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_METADATA_DIR,
    DEFAULT_OUTPUT_DIR,
    InvalidModelResponse,
    gemini_generate_content,
    is_retryable_api_error,
    iter_metadata_files,
    load_json,
    require_api_key,
    retry_call,
    write_json,
)


# gemini25flash
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the current user question based on the image and the prior conversation history."
)
DEFAULT_MODEL_NAME = "gemini-2.5-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Gemini API models on FPCO-Dialog image metadata."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Gemini model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help=f"Metadata json directory. Default: {DEFAULT_METADATA_DIR}",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Image directory. Default: {DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Subdirectory under output-dir. Default: normalized model name.",
    )
    parser.add_argument(
        "--existing-output",
        choices=("skip", "overwrite"),
        default="skip",
        help="When output json already exists, skip finished responses or overwrite them.",
    )
    parser.add_argument(
        "--question-field",
        choices=("modified", "content", "auto"),
        default="auto",
        help="Which question text to send to the model. auto prefers modified, then content.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for each answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 means greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p used only when temperature > 0.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=4,
        help="Max attempts per question on transient failures.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=8.0,
        help="Base wait seconds before retry. Backoff is applied automatically.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N json files after sorting.",
    )
    parser.add_argument(
        "--only-id",
        nargs="*",
        default=None,
        help="Only process selected file ids, e.g. 1101 1102.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of metadata files to process concurrently. Each file still runs its questions sequentially.",
    )
    return parser.parse_args()


def normalize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model_name.lower())


def read_image_as_base64(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return encoded


class GeminiRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.system_prompt = args.system_prompt
        self.api_key = require_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def _effective_max_output_tokens(self) -> int:
        model_name = self.model_name.lower()
        if "pro" in model_name:
            # Gemini Pro families can spend a large fraction of the output budget
            # on internal reasoning and otherwise terminate with MAX_TOKENS
            # before emitting any visible answer.
            return max(self.max_new_tokens, 1024)
        return self.max_new_tokens

    def generate(self, image_path: Path, history: list[dict[str, Any]]) -> str:
        contents = self._build_contents(history=history, image_path=image_path)
        payload = {
            "system_instruction": {
                "parts": [{"text": self.system_prompt}],
            },
            "contents": contents,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxOutputTokens": self._effective_max_output_tokens(),
            },
        }

        response_payload = self._post_generate_content(payload)
        return self._extract_text(response_payload)

    def _build_contents(self, history: list[dict[str, Any]], image_path: Path) -> list[dict[str, Any]]:
        image_b64 = read_image_as_base64(image_path)
        contents: list[dict[str, Any]] = []
        first_user_seen = False

        for message in history:
            role = message.get("role")
            if role == "system":
                continue

            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Each non-system history message must have non-empty string content.")

            parts: list[dict[str, Any]] = []
            if role == "user" and not first_user_seen:
                first_user_seen = True
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64,
                        }
                    }
                )

            parts.append({"text": content})

            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                raise ValueError(f"Unsupported role in history: {role}")

            contents.append(
                {
                    "role": gemini_role,
                    "parts": parts,
                }
            )

        if not first_user_seen:
            raise ValueError("Conversation history must include at least one user message.")

        return contents

    def _post_generate_content(self, payload: dict[str, Any]) -> dict[str, Any]:
        return gemini_generate_content(self.api_key, self.model_name, payload, 300.0)

    def _extract_text(self, response_payload: dict[str, Any]) -> str:
        candidates = response_payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            prompt_feedback = response_payload.get("promptFeedback")
            if isinstance(prompt_feedback, dict):
                block_reason = prompt_feedback.get("blockReason")
                if isinstance(block_reason, str) and block_reason.strip():
                    return f"[Gemini blocked this request: {block_reason}]"
            raise InvalidModelResponse(f"Gemini API returned no candidates: {response_payload}")

        first_candidate = candidates[0]
        content = first_candidate.get("content")
        if not isinstance(content, dict):
            finish_reason = first_candidate.get("finishReason")
            if isinstance(finish_reason, str) and finish_reason.strip():
                return f"[Gemini blocked this request: {finish_reason}]"
            raise InvalidModelResponse(f"Gemini candidate is missing content: {response_payload}")

        parts = content.get("parts")
        if not isinstance(parts, list):
            finish_reason = first_candidate.get("finishReason")
            if isinstance(finish_reason, str) and finish_reason.strip():
                return f"[Gemini blocked this request: {finish_reason}]"
            raise InvalidModelResponse(f"Gemini candidate content is missing parts: {response_payload}")

        text_chunks = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_chunks.append(part["text"])

        text = "".join(text_chunks).strip()
        if text:
            return text

        finish_reason = first_candidate.get("finishReason")
        prompt_feedback = response_payload.get("promptFeedback")
        raise InvalidModelResponse(
            "Gemini API returned empty text output. "
            f"finishReason={finish_reason} promptFeedback={prompt_feedback}"
        )


def retry_generate(
    runner: GeminiRunner,
    image_path: Path,
    history: list[dict[str, Any]],
    retry_attempts: int,
    retry_wait_seconds: float,
    file_id: str,
    question_id: Any,
) -> str:
    return retry_call(
        lambda: runner.generate(image_path=image_path, history=history),
        attempts=retry_attempts,
        wait_seconds=retry_wait_seconds,
        context=f"file={file_id} question={question_id}",
        retryable=is_retryable_api_error,
    )


def process_file(
    json_path: Path,
    image_dir: Path,
    output_path: Path,
    runner: GeminiRunner,
    args: argparse.Namespace,
) -> None:
    source_payload = load_json(json_path)
    existing_payload = load_json(output_path) if output_path.exists() else None
    output_payload = init_output_payload(
        source_payload=source_payload,
        existing_payload=existing_payload,
        existing_mode=args.existing_output,
    )

    metadata = output_payload["metadata"]

    question_items = output_payload["question"]

    image_id = str(metadata.get("id", json_path.stem))
    image_path = image_dir / f"{image_id}.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for {json_path.name}: {image_path}")

    if not output_path.exists():
        write_json(output_path, output_payload)

    for index, question_item in enumerate(question_items):
        question_id = question_item.get("id", index + 1)
        existing_response = question_item.get("response")
        if (
            args.existing_output == "skip"
            and isinstance(existing_response, str)
            and existing_response.strip()
        ):
            print(
                f"[skip] file={image_id} question={question_id} already has response.",
                flush=True,
            )
            continue

        history = build_history(
            question_items=question_items,
            current_index=index,
            question_field=args.question_field,
            system_prompt=args.system_prompt,
        )
        response = retry_generate(
            runner=runner,
            image_path=image_path,
            history=history,
            retry_attempts=args.retry_attempts,
            retry_wait_seconds=args.retry_wait_seconds,
            file_id=image_id,
            question_id=question_id,
        )
        question_item["response"] = response
        write_json(output_path, output_payload)
        print(
            f"[ok] file={image_id} question={question_id} saved to {output_path}",
            flush=True,
        )


def main() -> None:
    args = parse_args()

    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"Metadata dir not found: {args.metadata_dir}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")

    output_subdir = args.output_subdir or normalize_model_name(args.model_name)
    output_dir = args.output_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    only_ids = set(args.only_id) if args.only_id else None
    json_files = iter_metadata_files(args.metadata_dir, args.limit, only_ids)
    if not json_files:
        print("No metadata json files matched the current filters.", flush=True)
        return

    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1.")

    runner = GeminiRunner(args)
    worker_count = min(args.num_workers, len(json_files))
    print(
        f"Loaded model={args.model_name} output_dir={output_dir} num_workers={worker_count}",
        flush=True,
    )

    def run_one_file(json_path: Path) -> None:
        output_path = output_dir / json_path.name
        print(f"[start] {json_path.name}", flush=True)
        process_file(
            json_path=json_path,
            image_dir=args.image_dir,
            output_path=output_path,
            runner=runner,
            args=args,
        )

    if worker_count == 1:
        for json_path in json_files:
            run_one_file(json_path)
        return

    failures: list[tuple[Path, Exception]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_path = {
            executor.submit(run_one_file, json_path): json_path
            for json_path in json_files
        }
        for future in as_completed(future_to_path):
            json_path = future_to_path[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((json_path, exc))
                print(f"[failed] {json_path.name}: {exc}", flush=True)

    if failures:
        failed_names = ", ".join(path.name for path, _ in failures[:10])
        raise RuntimeError(
            f"{len(failures)} metadata files failed during benchmark. First failures: {failed_names}"
        )


if __name__ == "__main__":
    main()
