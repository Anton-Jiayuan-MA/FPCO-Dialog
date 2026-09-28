import argparse
import math
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_LAYER_COUNT = {
    "internvl2_5-1b": 24,
    "internvl2_5-2b": 24,
    "internvl2_5-4b": 36,
    "internvl2_5-8b": 32,
    "internvl2_5-26b": 48,
    "internvl2_5-38b": 64,
    "internvl2_5-78b": 80,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local InternVL models on FPCO-Dialog image metadata."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local InternVL model directory, e.g. models/InternVL2_5-8B",
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
        default="bfloat16",
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
        help="Use Transformers device_map. Prefer none for single-GPU efficiency.",
    )
    parser.add_argument(
        "--max-num-tiles",
        type=int,
        default=12,
        help="Maximum number of image tiles used by InternVL dynamic preprocessing.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=448,
        help="InternVL image tile size. Default: 448.",
    )
    parser.add_argument(
        "--use-flash-attn",
        choices=("auto", "true", "false"),
        default="auto",
        help="Whether to request flash attention when loading the model.",
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


def should_use_flash_attn(mode: str) -> bool:
    return mode == "true"


def resolve_visible_cuda_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def build_internvl_device_map(model_name: str, world_size: int) -> dict[str, int]:
    normalized_name = model_name.lower()
    if normalized_name not in MODEL_LAYER_COUNT:
        raise ValueError(f"Unsupported InternVL model for custom device map: {model_name}")
    if world_size < 2:
        raise ValueError("Custom InternVL device map requires at least 2 visible CUDA devices.")

    num_layers = MODEL_LAYER_COUNT[normalized_name]
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    layer_budget = [num_layers_per_gpu] * world_size
    layer_budget[0] = math.ceil(layer_budget[0] * 0.5)

    device_map: dict[str, int] = {}
    layer_index = 0
    for gpu_index, gpu_layers in enumerate(layer_budget):
        for _ in range(gpu_layers):
            if layer_index >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_index}"] = gpu_index
            layer_index += 1

    shared_modules = (
        "vision_model",
        "mlp1",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
        "language_model.output",
        "language_model.model.norm",
        "language_model.model.rotary_emb",
        "language_model.lm_head",
    )
    for module_name in shared_modules:
        device_map[module_name] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_path: Path, input_size: int, max_num_tiles: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num_tiles,
    )
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


class LocalInternRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        if transformers_version != "4.37.2":
            raise RuntimeError(
                "InternVL2.5 requires its separate Transformers 4.37.2 environment. "
                "Install requirements/requirements-internvl.txt in a new virtual environment; see code/README.md."
            )
        self.model_path = args.model_path
        self.device = args.device
        self.device_map = self._resolve_device_map(args.device_map)
        self.torch_dtype = resolve_dtype(args.torch_dtype)
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.system_prompt = args.system_prompt
        self.input_size = args.input_size
        self.max_num_tiles = args.max_num_tiles

        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
            "use_flash_attn": should_use_flash_attn(args.use_flash_attn),
        }
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
            # Transformers requires this when device_map is enabled.
            model_kwargs["low_cpu_mem_usage"] = True
        else:
            model_kwargs["low_cpu_mem_usage"] = False

        self.model = AutoModel.from_pretrained(self.model_path, **model_kwargs).eval()
        # "auto" is a loading option, not a valid dtype for image_tensor.to(...).
        self.torch_dtype = self.model.dtype
        self.model.system_message = self.system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        if self.device_map is None:
            self.model = self.model.to(self.device)

    def _resolve_device_map(self, device_map_mode: str) -> Any:
        if device_map_mode == "none":
            return None

        visible_cuda_count = resolve_visible_cuda_count()
        if visible_cuda_count <= 1:
            return None
        return build_internvl_device_map(self.model_path.name, visible_cuda_count)

    def _generation_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            config["temperature"] = self.temperature
            config["top_p"] = self.top_p
        return config

    def _pixel_device(self) -> str:
        if self.device_map is None:
            return self.device
        return "cuda:0"


def build_intern_inputs(history: list[dict[str, Any]], system_prompt: str) -> tuple[str, list[tuple[str, str]] | None]:
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    first_user_seen = False

    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each non-system history message must have non-empty string content.")

        if role == "user":
            user_text = content.strip()
            if not first_user_seen:
                first_user_seen = True
                user_text = f"<image>\n{user_text}"
            pending_user = user_text
        elif role == "assistant":
            if pending_user is None:
                raise ValueError("Assistant history message appeared before a user message.")
            pairs.append((pending_user, content.strip()))
            pending_user = None
        else:
            raise ValueError(f"Unsupported role in history: {role}")

    if pending_user is None:
        raise ValueError("History is missing the current user question.")

    current_question = pending_user
    if not pairs:
        return current_question, None
    return current_question, pairs


def generate_response(
    runner: LocalInternRunner,
    image_path: Path,
    history: list[dict[str, Any]],
    system_prompt: str,
) -> str:
    question, chat_history = build_intern_inputs(history, system_prompt)
    pixel_values = load_image(
        image_path=image_path,
        input_size=runner.input_size,
        max_num_tiles=runner.max_num_tiles,
    ).to(runner.torch_dtype)
    pixel_values = pixel_values.to(runner._pixel_device())
    with torch.inference_mode():
        response = runner.model.chat(
            runner.tokenizer,
            pixel_values,
            question,
            runner._generation_config(),
            history=chat_history,
            return_history=False,
        )
    return response.strip()


def process_file(
    json_path: Path,
    image_dir: Path,
    output_path: Path,
    runner: LocalInternRunner,
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
        response = generate_response(
            runner=runner,
            image_path=image_path,
            history=history,
            system_prompt=args.system_prompt,
        )
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

    runner = LocalInternRunner(args)
    print(
        f"Loaded model={args.model_path} output_dir={output_dir}",
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
