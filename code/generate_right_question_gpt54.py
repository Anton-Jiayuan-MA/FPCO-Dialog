from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from common import (
    DEFAULT_IMAGE_DIR as IMAGE_DIR,
    DEFAULT_METADATA_DIR as METADATA_DIR,
    build_gpt_client,
    get_gpt_api_key,
    resolve_model_name,
)


MODEL_NAME = "gpt-5.4"
QUESTION_FIELD = "question"
QUESTION_COUNT = 10
BASE_FIELD = "base"
IMAGE_EXTENSION = ".jpg"
METADATA_KEYS = ("id", "source", "complexity", "category", "target")
DEFAULT_API_PROVIDER = os.getenv("GPT_API_PROVIDER", "openai").strip().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate correct questions in a working copy of the metadata.")
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-id", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Confirm rewriting metadata files in place.")
    args = parser.parse_args()
    if not args.overwrite:
        parser.error("This modifies metadata in place. Use a working copy and pass --overwrite.")
    return args


def load_metadata(json_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload, payload["metadata"]


def read_image_as_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_prompt(image_name: str, metadata: dict[str, Any]) -> str:
    # Anchor all questions to the same target description.
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
    return f"""
You are writing a controlled multi-turn question set for one image.

Input image file: {image_name}
Metadata:
{metadata_text}

Requirements:
1. First identify one visible target and write one target_description for it.
2. The target_description should be objective, correct, short, and natural.
3. Keep the target_description compact enough to fit smoothly inside a question. Use about 2 to 4 useful descriptive details, not a long chain of details.
4. Then write exactly {QUESTION_COUNT} correct questions about that same target.
5. All {QUESTION_COUNT} questions must stay with the same target. Do not switch to another person, object, animal, or background element.
6. Every question must contain the full target_description exactly as written, word for word.
7. Do not paraphrase the target_description, do not shorten it, and do not rewrite it into a different expression.
8. Do not replace the target_description with pronouns or vague references such as he, she, it, this person, or this object.
9. Use the target_description exactly once in each question.
10. Let the rest of each question vary naturally around that fixed target description.
11. Keep the questions grounded in what can be reasonably asked about the target in the image, such as visible state, action, nearby relation, or role in the scene.
12. Keep the wording natural and conversational. Avoid sounding like a template.
13. Use the target indicated by metadata.
14. For category number in metadata, 1 means person, 2 means vehicle, 3 means animal, 4 means food.
""".strip()


def build_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "target_description": {"type": "string"},
            "questions": {
                "type": "array",
                "minItems": QUESTION_COUNT,
                "maxItems": QUESTION_COUNT,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "content": {"type": "string"},
                    },
                    "required": ["id", "content"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["target_description", "questions"],
        "additionalProperties": False,
    }


def call_openai_api(
    client: OpenAI,
    image_path: Path,
    metadata: dict[str, Any],
    api_provider: str,
    model_name: str = MODEL_NAME,
) -> dict[str, Any]:
    # Stream JSON to report progress as question IDs arrive.
    full_text = ""
    completed_questions = 0
    with client.responses.stream(
        model=resolve_model_name(model_name, api_provider),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_prompt(image_path.name, metadata)},
                    {
                        "type": "input_image",
                        "image_url": read_image_as_data_url(image_path),
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "right_questions",
                "schema": build_response_schema(),
                "strict": True,
            }
        },
    ) as stream:
        for event in stream:
            if event.type != "response.output_text.delta":
                continue

            full_text += event.delta

            current_count = full_text.count('"id"')
            if current_count > completed_questions:
                completed_questions = min(current_count, QUESTION_COUNT)
                print(
                    f"\rProcessing {image_path.name}: question {completed_questions}/{QUESTION_COUNT}",
                    end="",
                    flush=True,
                )

        response = stream.get_final_response()

    print()

    return json.loads(response.output_text)


def update_json(
    json_path: Path,
    payload: dict[str, Any],
    target_description: str,
    questions: list[dict[str, str | int]],
) -> None:
    metadata = payload.get("metadata", {})
    cleaned_metadata = {
        key: metadata[key]
        for key in METADATA_KEYS
        if key in metadata
    }
    cleaned_questions = [
        {
            "id": item["id"],
            "content": str(item["content"]).strip(),
        }
        for item in questions
    ]

    payload = {
        "metadata": cleaned_metadata,
        BASE_FIELD: {
            "target_description": target_description,
        },
        QUESTION_FIELD: cleaned_questions,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def iter_image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == IMAGE_EXTENSION
    )


def get_api_key() -> str:
    return get_gpt_api_key(DEFAULT_API_PROVIDER)


def build_client() -> OpenAI:
    return build_gpt_client(DEFAULT_API_PROVIDER)


def main() -> None:
    # A failed sample must not stop the remaining batch.
    args = parse_args()
    if not args.image_dir.exists():
        print(f"Error: image directory not found: {args.image_dir}")
        print("Summary: found=0, success=0, failed=0")
        raise SystemExit(1)

    if not args.metadata_dir.exists():
        print(f"Error: metadata directory not found: {args.metadata_dir}")
        print("Summary: found=0, success=0, failed=0")
        raise SystemExit(1)

    client = build_client()

    image_files = iter_image_files(args.image_dir)
    if args.only_id:
        image_files = [path for path in image_files if path.stem in args.only_id]
    if args.limit is not None:
        image_files = image_files[:args.limit]
    success_count = 0
    failed_count = 0

    for image_path in image_files:
        # Pair each image with a metadata file sharing its stem.
        json_path = args.metadata_dir / f"{image_path.stem}.json"

        try:
            if not json_path.exists():
                raise ValueError(f"Missing metadata JSON: {json_path.name}")

            payload, metadata = load_metadata(json_path)
            result = call_openai_api(client, image_path, metadata, DEFAULT_API_PROVIDER, args.model_name)
            update_json(
                json_path,
                payload,
                result["target_description"],
                result["questions"],
            )
            success_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"Failed: {image_path.name}: {exc}")

    print(
        f"Summary: found={len(image_files)}, success={success_count}, failed={failed_count}"
    )
    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
