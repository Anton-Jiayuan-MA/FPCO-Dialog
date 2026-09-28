import argparse
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import APIConnectionError, OpenAI

from benchmark_common import build_history, init_output_payload
from common import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_METADATA_DIR,
    DEFAULT_OUTPUT_DIR,
    InvalidModelResponse,
    is_retryable_api_error,
    iter_metadata_files,
    load_json,
    require_api_key,
    retry_call,
    write_json,
)


QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the current user question based on the image and the prior conversation history."
)
DEFAULT_MODEL_NAME = "qwen-vl-max"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen API models on FPCO-Dialog image metadata."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Qwen API model name. Default: {DEFAULT_MODEL_NAME}",
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
        help="Sampling temperature. 0 means deterministic decoding where supported.",
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
        "--num-worker",
        type=int,
        default=1,
        help="Number of parallel image workers. Questions within one image are always processed sequentially.",
    )
    return parser.parse_args()


def normalize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model_name.lower())


def read_image_as_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class QwenAPIRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.system_prompt = args.system_prompt
        self.client = OpenAI(
            api_key=require_api_key("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
            base_url=QWEN_BASE_URL,
            max_retries=0,
        )

    def generate(self, image_path: Path, history: list[dict[str, Any]]) -> str:
        messages = self._build_messages(history=history, image_path=image_path)
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
        }
        if self.temperature > 0:
            request_kwargs["temperature"] = self.temperature
            request_kwargs["top_p"] = self.top_p

        response = self.client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        message = choice.message
        text = (message.content or "").strip()
        if text:
            return text
        raise InvalidModelResponse(f"Qwen API returned empty output for model={self.model_name}.")

    def _build_messages(self, history: list[dict[str, Any]], image_path: Path) -> list[dict[str, Any]]:
        image_url = read_image_as_data_url(image_path)
        messages: list[dict[str, Any]] = []
        first_user_seen = False

        for message in history:
            role = message.get("role")
            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError("Each history message must have non-empty string content.")

            if role == "system":
                messages.append({"role": "system", "content": content})
                continue

            parts: list[dict[str, Any]] = []
            if role == "user" and not first_user_seen:
                first_user_seen = True
                parts.append({"type": "image_url", "image_url": {"url": image_url}})

            if role == "user":
                parts.append({"type": "text", "text": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
                continue
            else:
                raise ValueError(f"Unsupported role in history: {role}")

            messages.append({"role": role, "content": parts})

        if not first_user_seen:
            raise ValueError("Conversation history must include at least one user message.")

        return messages


def retry_generate(
    runner: QwenAPIRunner,
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
        retryable=lambda exc: is_retryable_api_error(exc, (APIConnectionError,)),
    )


def process_file(
    json_path: Path,
    image_dir: Path,
    output_path: Path,
    runner: QwenAPIRunner,
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


def process_file_worker(
    json_path: Path,
    image_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> str:
    runner = QwenAPIRunner(args)
    output_path = output_dir / json_path.name
    print(f"[start] {json_path.name}", flush=True)
    process_file(
        json_path=json_path,
        image_dir=image_dir,
        output_path=output_path,
        runner=runner,
        args=args,
    )
    return json_path.name


def main() -> None:
    args = parse_args()

    if args.num_worker < 1:
        raise ValueError(f"--num-worker must be >= 1, got {args.num_worker}")
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

    num_worker = min(args.num_worker, len(json_files))
    print(
        f"Loaded model={args.model_name} output_dir={output_dir} num_worker={num_worker}",
        flush=True,
    )

    if num_worker == 1:
        runner = QwenAPIRunner(args)
        for json_path in json_files:
            output_path = output_dir / json_path.name
            print(f"[start] {json_path.name}", flush=True)
            process_file(
                json_path=json_path,
                image_dir=args.image_dir,
                output_path=output_path,
                runner=runner,
                args=args,
            )
        return

    failures: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        future_to_file = {
            executor.submit(
                process_file_worker,
                json_path=json_path,
                image_dir=args.image_dir,
                output_dir=output_dir,
                args=args,
            ): json_path.name
            for json_path in json_files
        }

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                finished_file = future.result()
            except Exception as exc:
                failures.append((file_name, exc))
                print(f"[error] {file_name}: {exc}", flush=True)
            else:
                print(f"[done] {finished_file}", flush=True)

    if failures:
        summary = "; ".join(f"{file_name}: {exc}" for file_name, exc in failures)
        raise RuntimeError(f"{len(failures)} file(s) failed: {summary}")


if __name__ == "__main__":
    main()
