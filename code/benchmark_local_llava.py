import argparse
import copy
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local LLaVA-OneVision models on FPCO-Dialog image metadata."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local LLaVA model directory, e.g. models/LLaVA-OneVision-7B",
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
        help="Use Transformers device_map. auto is useful for larger models.",
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
        "--max-question-id",
        type=int,
        default=None,
        help="Only run questions whose id is <= this value.",
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


class LocalLlavaRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_path = args.model_path
        self.device = args.device
        self.device_map = None if args.device_map == "none" else args.device_map
        self.torch_dtype = resolve_dtype(args.torch_dtype)
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.system_prompt = args.system_prompt

        self._prime_cuda_if_needed()

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.model_type = getattr(config, "model_type", "")
        if self.model_type == "llava_next":
            self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
            self._normalize_llava_next_processor(config)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map=self.device_map,
            )

        if self.device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        self._enable_cache()
        self.input_device = self._resolve_input_device()

    def _normalize_llava_next_processor(self, config: Any) -> None:
        vision_config = getattr(config, "vision_config", None)

        if getattr(self.processor, "patch_size", None) is None and vision_config is not None:
            patch_size = getattr(vision_config, "patch_size", None)
            if patch_size is not None:
                self.processor.patch_size = patch_size

        if getattr(self.processor, "vision_feature_select_strategy", None) is None:
            feature_select_strategy = getattr(config, "vision_feature_select_strategy", None)
            if feature_select_strategy is not None:
                self.processor.vision_feature_select_strategy = feature_select_strategy

        if getattr(self.processor, "num_additional_image_tokens", 0) == 0:
            # CLIP-like vision towers used by LLaVA-NeXT include an extra CLS token.
            # Without this, the processor underestimates image tokens by exactly one.
            self.processor.num_additional_image_tokens = 1

    def _prime_cuda_if_needed(self) -> None:
        target_device = None
        if self.device_map is None and str(self.device).startswith("cuda"):
            target_device = self.device
        elif self.device_map == "auto" and torch.cuda.is_available():
            target_device = "cuda:0"

        if target_device is None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA requested for LLaVA benchmark but no visible GPU is available (device={target_device})."
            )

        # Force CUDA context initialization before model loading. Without this,
        # some OneVision checkpoints intermittently fail later at model.to(...).
        torch.empty(1, device=target_device)

    def _generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
        }
        if self.temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _resolve_input_device(self) -> str:
        if self.device_map is None:
            return self.device

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for mapped_device in hf_device_map.values():
                if isinstance(mapped_device, int):
                    return f"cuda:{mapped_device}"
                if isinstance(mapped_device, torch.device):
                    if mapped_device.type != "cpu":
                        return str(mapped_device)
                if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk"}:
                    if mapped_device.isdigit():
                        return f"cuda:{mapped_device}"
                    return mapped_device

        if torch.cuda.is_available():
            return "cuda:0"
        return self.device

    def _enable_cache(self) -> None:
        # Enable KV caching on the outer and nested text configurations.
        for config_like in (
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "config", None), "text_config", None),
            getattr(self.model, "generation_config", None),
        ):
            if config_like is not None and hasattr(config_like, "use_cache"):
                config_like.use_cache = True

    def _move_inputs(self, inputs: Any) -> Any:
        return inputs.to(self.input_device)

    def _build_llava_messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages = copy.deepcopy(history)
        first_user_seen = False

        # LLaVA-OneVision uses list-based multimodal content.
        # Attach the image only to the first user turn.
        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if isinstance(content, str):
                content_items = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                content_items = content
            else:
                raise ValueError("Unsupported message content format for LLaVA chat template.")

            if role == "user" and not first_user_seen:
                first_user_seen = True
                has_image = any(
                    isinstance(item, dict) and item.get("type") == "image"
                    for item in content_items
                )
                if not has_image:
                    content_items.insert(0, {"type": "image"})

            message["content"] = content_items

        if not first_user_seen:
            raise ValueError("Conversation history must include at least one user message.")
        return messages

    def generate(self, image_path: Path, history: list[dict[str, Any]]) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = self._build_llava_messages(history)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=prompt,
            images=[image],
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
    runner: LocalLlavaRunner,
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
        if args.max_question_id is not None and int(question_id) > args.max_question_id:
            break

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
        response = runner.generate(image_path=image_path, history=history)
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

    runner = LocalLlavaRunner(args)
    print(
        f"Loaded model={args.model_path} model_type={runner.model_type} output_dir={output_dir}",
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
