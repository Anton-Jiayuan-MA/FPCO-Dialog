import argparse
import copy
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

from benchmark_common import build_history, init_output_payload
from common import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_METADATA_DIR,
    DEFAULT_OUTPUT_DIR,
    iter_metadata_files,
    load_json,
    write_json,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the current user question based on the image and the prior conversation history."
)
VISION_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local Qwen-family models on FPCO-Dialog image metadata."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local Qwen model directory, e.g. models/Qwen2.5-VL-3B-Instruct",
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
        help="Subdirectory under output-dir. Default: normalized model directory name.",
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
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device when --device-map none, e.g. "cuda", "cuda:0", "cpu".',
    )
    parser.add_argument(
        "--device-map",
        choices=("none", "auto"),
        default="none",
        help="Use Transformers device_map. auto is useful for larger Qwen models.",
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
    return parser.parse_args()


def normalize_model_name(model_path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "", model_path.name.lower())


def resolve_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


class LocalQwenRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_path = args.model_path
        self.device = args.device
        self.device_map = None if args.device_map == "none" else args.device_map
        self.torch_dtype = resolve_dtype(args.torch_dtype)
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.system_prompt = args.system_prompt

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.model_type = getattr(config, "model_type", "")
        self.is_vision = self.model_type in VISION_MODEL_TYPES

        if self.is_vision:
            self.processor = self._load_processor()
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map=self.device_map,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map=self.device_map,
            )

        if self.device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _load_processor(self) -> Any:
        try:
            return AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        except (ValueError, TypeError) as exc:
            if self.model_type == "qwen2_5_vl":
                tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                image_processor = Qwen2VLImageProcessor.from_pretrained(self.model_path)
                video_processor = Qwen2VLVideoProcessor.from_pretrained(self.model_path)
                return Qwen2_5_VLProcessor(
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                    video_processor=video_processor,
                    chat_template=tokenizer.chat_template,
                )
            raise RuntimeError(
                f"Failed to load processor for model_type={self.model_type} from {self.model_path}"
            ) from exc

    def _generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
        }
        if self.temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _move_inputs(self, inputs: Any) -> Any:
        if self.device_map is None:
            return inputs.to(self.device)
        return inputs

    def generate(self, image_path: Path | None, history: list[dict[str, Any]]) -> str:
        if self.is_vision:
            if image_path is None:
                raise ValueError("Vision Qwen models require an image path.")
            return self._generate_vision(image_path=image_path, history=history)
        return self._generate_text(history=history)

    def _generate_text(self, history: list[dict[str, Any]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = self._move_inputs(inputs)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self._generation_kwargs())

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _generate_vision(self, image_path: Path, history: list[dict[str, Any]]) -> str:
        patched_history = copy.deepcopy(history)
        image = Image.open(image_path).convert("RGB")
        first_user_seen = False

        # Normalize every message into multimodal list format because
        # the current processor implementation expects list-based content
        # for all messages when tokenize=True.
        for message in patched_history:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                raise ValueError("Unsupported message content format for vision chat template.")

            if message.get("role") == "user" and not first_user_seen:
                first_user_seen = True
                has_image = any(
                    isinstance(item, dict) and item.get("type") == "image"
                    for item in message["content"]
                )
                if not has_image:
                    message["content"].insert(0, {"type": "image", "image": image})

        if not first_user_seen:
            raise ValueError("Conversation history must include at least one user message.")

        inputs = self.processor.apply_chat_template(
            patched_history,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self._generation_kwargs())

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, prompt_length:]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def process_file(
    json_path: Path,
    image_dir: Path,
    output_path: Path,
    runner: LocalQwenRunner,
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
    if runner.is_vision and not image_path.exists():
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
        response = runner.generate(image_path=image_path if runner.is_vision else None, history=history)
        question_item["response"] = response
        write_json(output_path, output_payload)
        print(
            f"[ok] file={image_id} question={question_id} saved to {output_path}",
            flush=True,
        )


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"Metadata dir not found: {args.metadata_dir}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")

    output_subdir = args.output_subdir or normalize_model_name(args.model_path)
    output_dir = args.output_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    only_ids = set(args.only_id) if args.only_id else None
    json_files = iter_metadata_files(args.metadata_dir, args.limit, only_ids)
    if not json_files:
        print("No metadata json files matched the current filters.", flush=True)
        return

    runner = LocalQwenRunner(args)
    print(
        f"Loaded model={args.model_path} model_type={runner.model_type} "
        f"is_vision={runner.is_vision} output_dir={output_dir}",
        flush=True,
    )

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


if __name__ == "__main__":
    main()
