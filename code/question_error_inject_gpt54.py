from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from common import (
    DEFAULT_METADATA_DIR as METADATA_DIR,
    build_gpt_client,
    get_gpt_api_key,
    load_json,
    resolve_model_name,
)


MODEL_NAME = "gpt-5.4"
QUESTION_FIELD = "question"
QUESTION_COUNT = 10
BASE_FIELD = "base"
METADATA_KEYS = ("id", "source", "complexity", "category", "target")
DEFAULT_API_PROVIDER = os.getenv("GPT_API_PROVIDER", "openai").strip().lower()

FALSE_IDENTITY = "identity"
FALSE_ATTRIBUTE = "attribute"
FALSE_LOCATION = "location"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject false premises into FPCO-Dialog metadata files."
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=METADATA_DIR,
        help=f"Metadata json directory. Default: {METADATA_DIR}",
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
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--overwrite", action="store_true", help="Confirm rewriting metadata files in place.")
    args = parser.parse_args()
    if not args.overwrite:
        parser.error("This modifies metadata in place. Use a working copy and pass --overwrite.")
    return args


def save_json(json_path: Path, payload: dict[str, Any]) -> None:
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def assign_false_premises(questions: list[dict[str, Any]]) -> dict[int, bool]:
    # Keep the first three turns correct and inject false premises into turns 4-10.
    if len(questions) != QUESTION_COUNT:
        raise ValueError(f"Expected {QUESTION_COUNT} questions, got {len(questions)}")

    question_ids = [item.get("id") for item in questions]
    if question_ids != list(range(1, QUESTION_COUNT + 1)):
        raise ValueError(f"Question ids must be exactly 1..{QUESTION_COUNT} in order")

    return {question_id: question_id >= 4 for question_id in question_ids}


def infer_false_class_from_file_id(file_id: str) -> str:
    # The final two ID digits determine the false-premise category.
    image_index = int(str(file_id)[-2:])
    if 1 <= image_index <= 30:
        return FALSE_IDENTITY
    if 31 <= image_index <= 60:
        return FALSE_ATTRIBUTE
    if 61 <= image_index <= 90:
        return FALSE_LOCATION
    raise ValueError(f"Unsupported image index for false class mapping: {file_id}")


def build_injection_prompt(
    metadata: dict[str, Any],
    target_description: str,
    false_class: str,
    false_items: list[dict[str, Any]],
) -> str:
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
    items_text = json.dumps(false_items, ensure_ascii=False, indent=2)

    return f"""
You are creating controlled false-premise questions for one image.

Metadata:
{metadata_text}

Original target_description:
{target_description}

False-premise questions to modify:
{items_text}

Think through the edit internally, but only return the final JSON.

The false class for this image is fixed:
{false_class}

Requirements:
1. First choose one modified_target_description based on the original target_description.
2. Change only one word if possible. If one word is not natural, change only one short local phrase.
3. Keep the rest of the target description unchanged.
4. Do not expand the description.
5. Do not create multiple errors.
6. The result should still sound like a natural referring expression.
7. The changed part must match the fixed false class.
8. For identity edits, use a clearly incompatible identity substitution. Avoid age-graded, near-synonym, vague, or broader/narrower label changes such as woman -> girl, man -> boy, or person -> woman. Prefer a firmer identity change such as woman <-> man, dog <-> cat, or car <-> truck when it fits naturally. Also avoid leaving an obvious gender-pronoun mismatch in the final questions; if needed, choose a natural compatible identity edit or a neutral expression that still changes the target identity.
9. For location edits, make the change a natural position or relative-position change of the whole target. The modified_target_description must remain a complete referring expression that can be placed directly into a question. Attach the location relation to the target and a scene place or scene object, not to clothing, attributes, body parts, or local descriptive phrases. Do not treat clothing phrases such as "in a green shirt" as scene locations. Prefer natural relations such as near, by, beside, next to, in front of, in the foreground, in the background, on the left, or on the right when they fit. Avoid unnatural attachments such as "behind a green shirt", "under dark hair", or "beside glasses".
10. For each question, replace the original target_description with the same modified_target_description.
11. Keep each question's intent and wording otherwise as unchanged as possible.
12. If an identity edit creates an obvious he/she/his/her mismatch, adjust only those pronouns to match the new identity or use a natural neutral phrasing.
13. Do not rewrite questions for style.

False class definitions:
- identity: change who or what the target is with a clearly incompatible identity, not a soft label shift
- attribute: change one property of the target
- location: change where the whole target is or its relative position in the scene

This modified_target_description will be reused across multiple turns for the same image, so keep it stable and concise.

Return JSON with:
- modified_target_description
- questions: an array of objects with id and modified
""".strip()


def build_rewritten_question_item(
    source_item: dict[str, Any],
    modified: str,
    false_premise: bool,
) -> dict[str, Any]:
    return {
        "id": source_item["id"],
        "content": str(source_item["content"]).strip(),
        "modified": modified,
        "false_premise": false_premise,
    }


def build_updated_base(
    target_description: str,
    modified_target_description: str,
    false_class: str,
) -> dict[str, Any]:
    # Store descriptions shared by every question in this sample.
    return {
        "target_description": target_description,
        "modified_target_description": modified_target_description,
        "false_class": false_class,
    }


def call_openai_for_injection(
    client: OpenAI,
    metadata: dict[str, Any],
    target_description: str,
    false_class: str,
    false_items: list[dict[str, Any]],
    api_provider: str,
    model_name: str = MODEL_NAME,
) -> tuple[str, dict[int, str]]:
    response = client.responses.create(
        model=resolve_model_name(model_name, api_provider),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_injection_prompt(
                            metadata,
                            target_description,
                            false_class,
                            false_items,
                        ),
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "false_premise_injection",
                "schema": {
                    "type": "object",
                    "properties": {
                        "modified_target_description": {"type": "string"},
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "modified": {"type": "string"},
                                },
                                "required": ["id", "modified"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["modified_target_description", "questions"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        },
    )

    rewritten = json.loads(response.output_text)
    modified_target_description = rewritten["modified_target_description"].strip()
    if not modified_target_description:
        raise ValueError("Model returned an empty modified_target_description")

    modified_questions = {
        int(item["id"]): str(item["modified"]).strip()
        for item in rewritten["questions"]
    }
    return modified_target_description, modified_questions


def process_single_file(
    client: OpenAI,
    json_path: Path,
    false_class: str,
    model_name: str = MODEL_NAME,
) -> None:
    # Write back only after all rewritten questions have been validated.
    payload = load_json(json_path)

    metadata = payload.get("metadata")
    base = payload.get(BASE_FIELD)
    questions = payload.get(QUESTION_FIELD)

    if not isinstance(metadata, dict):
        raise ValueError("Missing or invalid metadata")
    if not isinstance(base, dict):
        raise ValueError("Missing or invalid base")
    if not isinstance(questions, list):
        raise ValueError("Missing or invalid question list")
    target_description = str(base.get("target_description", "")).strip()
    if not target_description:
        raise ValueError("Missing target_description in base")

    assignments = assign_false_premises(questions)
    false_items = [
        {
            "id": item["id"],
            "content": item["content"].strip(),
        }
        for item in questions
        if assignments.get(item.get("id")) and isinstance(item.get("content"), str)
    ]
    modified_target_description, modified_question_map = call_openai_for_injection(
        client=client,
        metadata=metadata,
        target_description=target_description,
        false_class=false_class,
        false_items=false_items,
        api_provider=DEFAULT_API_PROVIDER,
        model_name=model_name,
    )

    rewritten_questions: list[dict[str, Any]] = []

    for item in questions:
        question_id = item.get("id")
        original_question = item.get("content")

        if not isinstance(question_id, int):
            raise ValueError("Question id must be an integer")
        if not isinstance(original_question, str) or not original_question.strip():
            raise ValueError(f"Question {question_id} has invalid content")

        false_premise = assignments[question_id]
        if false_premise:
            modified_question = modified_question_map.get(question_id, "")
            if not modified_question:
                raise ValueError(f"Missing modified question for question {question_id}")
        else:
            modified_question = original_question

        rewritten_questions.append(
            build_rewritten_question_item(
                source_item=item,
                modified=modified_question.strip(),
                false_premise=false_premise,
            )
        )

    cleaned_metadata = {
        key: metadata[key]
        for key in METADATA_KEYS
        if key in metadata
    }
    payload = {
        "metadata": cleaned_metadata,
        BASE_FIELD: build_updated_base(
            target_description=target_description,
            modified_target_description=modified_target_description,
            false_class=false_class,
        ),
        QUESTION_FIELD: rewritten_questions,
    }
    save_json(json_path, payload)


def iter_json_files(
    metadata_dir: Path,
    limit: int | None,
    only_ids: set[str] | None,
) -> list[Path]:
    files = sorted(
        path
        for path in metadata_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )
    if only_ids:
        files = [path for path in files if path.stem in only_ids]
    if limit is not None:
        files = files[:limit]
    return files


def get_api_key() -> str:
    return get_gpt_api_key(DEFAULT_API_PROVIDER)


def build_client() -> OpenAI:
    return build_gpt_client(DEFAULT_API_PROVIDER)


def main() -> None:
    args = parse_args()
    metadata_dir = args.metadata_dir
    only_ids = set(args.only_id) if args.only_id else None

    if not metadata_dir.exists():
        print(f"Error: metadata directory not found: {metadata_dir}")
        print("Summary: found=0, success=0, failed=0")
        raise SystemExit(1)

    client = build_client()

    json_files = iter_json_files(metadata_dir, args.limit, only_ids)
    success_count = 0
    failed_count = 0

    for json_path in json_files:
        try:
            process_single_file(
                client=client,
                json_path=json_path,
                false_class=infer_false_class_from_file_id(json_path.stem),
                model_name=args.model_name,
            )
            success_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"Failed: {json_path.name}: {exc}")

    print(
        f"Summary: found={len(json_files)}, success={success_count}, failed={failed_count}"
    )
    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
