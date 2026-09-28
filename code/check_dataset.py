"""Read-only validation of the published image/question pairs; no API key needed."""

import argparse
from pathlib import Path
from typing import Any

from common import DEFAULT_IMAGE_DIR, DEFAULT_METADATA_DIR, load_json


def validate_sample(payload: dict[str, Any], file_id: str) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or str(metadata.get("id")) != file_id:
        raise ValueError("metadata.id must match the file name")
    base = payload.get("base")
    if not isinstance(base, dict) or base.get("false_class") not in {"identity", "attribute", "location"}:
        raise ValueError("base.false_class must be identity, attribute, or location")
    for name in ("target_description", "modified_target_description"):
        if not isinstance(base.get(name), str) or not base[name].strip():
            raise ValueError(f"base.{name} must be non-empty text")
    questions = payload.get("question")
    if not isinstance(questions, list) or len(questions) != 10:
        raise ValueError("expected exactly 10 questions")
    for index, question in enumerate(questions, 1):
        if not isinstance(question, dict) or type(question.get("id")) is not int or question["id"] != index:
            raise ValueError("question IDs must be integers 1..10 in order")
        for name in ("content", "modified"):
            if not isinstance(question.get(name), str) or not question[name].strip():
                raise ValueError(f"question {index}: {name} must be non-empty text")
        if question.get("false_premise") is not (index >= 4):
            raise ValueError(f"question {index}: incorrect false_premise flag")


def validate_dataset(image_dir: Path, metadata_dir: Path, expected_count: int | None = None) -> int:
    if not image_dir.is_dir() or not metadata_dir.is_dir():
        raise ValueError("Data directories not found; use --image-dir and --metadata-dir to select them.")
    images = {path.stem for path in image_dir.glob("*.jpg") if path.is_file()}
    metadata_files = sorted(metadata_dir.glob("*.json"))
    ids = {path.stem for path in metadata_files}
    if not ids or ids != images:
        raise ValueError(
            f"Unpaired or empty dataset: missing images={len(ids - images)}, "
            f"missing metadata={len(images - ids)}, paired={len(ids & images)}"
        )
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} pairs, found {len(ids)}")
    for path in metadata_files:
        try:
            validate_sample(load_json(path), path.stem)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()
    count = validate_dataset(args.image_dir, args.metadata_dir, args.expected_count)
    print(f"OK: {count} paired images/metadata files and {count * 10} valid questions.")


if __name__ == "__main__":
    main()
