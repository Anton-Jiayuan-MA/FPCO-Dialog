from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any

import detection_common
from common import (
    DEFAULT_OUTPUT_DIR,
    InvalidModelResponse,
    gemini_generate_content,
    is_retryable_api_error,
    iter_output_subdirs,
    load_json,
    require_api_key,
    retry_call,
    write_json,
)


DEFAULT_DETECTOR_MODEL = "gemini-3.1-pro-preview"
QUESTION_FIELD = "question"
BASE_FIELD = "base"
DETECT_FIELD = "gemini31pro_detect"


should_skip_file = partial(detection_common.should_skip_file, detect_field=DETECT_FIELD)
collect_questions_for_detection = partial(detection_common.collect_questions_for_detection, detect_field=DETECT_FIELD)
build_detection_prompt = detection_common.build_detection_prompt
build_updated_question_item = partial(detection_common.build_updated_question_item, detect_field=DETECT_FIELD)
update_payload_with_labels = partial(detection_common.update_payload_with_labels, detect_field=DETECT_FIELD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect correction behavior with Gemini 3.1 Pro and write gemini31pro_detect into output JSON files."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_DETECTOR_MODEL,
        help=f"Gemini detector model name. Default: {DEFAULT_DETECTOR_MODEL}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-subdir",
        nargs="*",
        default=None,
        help="Only process selected output subdirectories.",
    )
    parser.add_argument(
        "--existing-detect",
        choices=("skip", "overwrite"),
        default="skip",
        help=f"Skip questions that already have {DETECT_FIELD}, or overwrite them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N json files per output subdirectory after sorting.",
    )
    parser.add_argument(
        "--only-id",
        nargs="*",
        default=None,
        help="Only process selected file ids, e.g. 1101 1102.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=4,
        help="Max detector attempts per file on transient failures.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=8.0,
        help="Base wait seconds before retry. Backoff is applied automatically.",
    )
    parser.add_argument(
        "--http-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for each Gemini API request in seconds.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of JSON files to process concurrently. Default: 1.",
    )
    return parser.parse_args()


def iter_json_files(output_subdir: Path, limit: int | None, only_ids: set[str] | None) -> list[Path]:
    files = sorted(
        path
        for path in output_subdir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )
    if only_ids:
        files = [path for path in files if path.stem in only_ids]
    if limit is not None:
        files = files[:limit]
    return files


def post_generate_content(
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = dict(payload)
    model_name = body.pop("model_name")
    return gemini_generate_content(api_key, model_name, body, timeout_seconds)


def extract_text(response_payload: dict[str, Any]) -> str:
    candidates = response_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        prompt_feedback = response_payload.get("promptFeedback")
        raise InvalidModelResponse(f"Gemini API returned no candidates: promptFeedback={prompt_feedback}")

    first_candidate = candidates[0]
    content = first_candidate.get("content")
    if not isinstance(content, dict):
        raise InvalidModelResponse("Gemini candidate is missing content.")

    parts = content.get("parts")
    if not isinstance(parts, list):
        raise InvalidModelResponse("Gemini candidate content is missing parts.")

    text_chunks = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_chunks.append(part["text"])

    text = "".join(text_chunks).strip()
    if text:
        return text

    finish_reason = first_candidate.get("finishReason")
    raise InvalidModelResponse(f"Gemini API returned empty text output. finishReason={finish_reason}")


def parse_detector_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    parsed = json.loads(cleaned)
    if isinstance(parsed, list):
        return {"labels": parsed}
    return parsed


def coerce_detect_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"Unsupported integer detect value: {value}")

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
        raise ValueError(f"Unsupported string detect value: {value!r}")

    raise ValueError(f"Unsupported detect value type: {type(value).__name__}")


def call_detector(
    api_key: str,
    model_name: str,
    file_id: str,
    false_class: str,
    questions: list[dict[str, Any]],
    timeout_seconds: float,
) -> dict[int, bool]:
    payload = {
        "model_name": model_name,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": build_detection_prompt(file_id, false_class, questions)}],
            }
        ],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
        "generationConfig": {
            "temperature": 0.0,
            "topP": 1.0,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }

    response_payload = post_generate_content(api_key, payload, timeout_seconds)
    payload_json = parse_detector_json(extract_text(response_payload))
    if not isinstance(payload_json, dict):
        raise InvalidModelResponse("Detector output must be a JSON object.")
    labels = payload_json.get("labels")
    if not isinstance(labels, list):
        raise InvalidModelResponse("Detector output is missing labels.")

    result: dict[int, bool] = {}
    expected_ids = {item["id"] for item in questions}

    for item in labels:
        if not isinstance(item, dict):
            raise InvalidModelResponse("Each detector label must be an object.")
        question_id = item.get("id")
        raw_detect = item.get("detect")
        if raw_detect is None and "label" in item:
            raw_detect = item.get("label")
        if not isinstance(question_id, int):
            raise InvalidModelResponse("Detector returned a non-integer question id.")
        try:
            detect = coerce_detect_value(raw_detect)
        except ValueError:
            raise InvalidModelResponse(f"Detector returned invalid detect for question {question_id}.")
        if question_id in result:
            raise InvalidModelResponse(f"Detector returned duplicate question id {question_id}.")
        result[question_id] = detect

    if set(result) != expected_ids:
        raise InvalidModelResponse(
            f"Detector returned mismatched ids. expected={sorted(expected_ids)} got={sorted(result)}"
        )

    return result


def get_api_key() -> str:
    return require_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")


def detect_single_file(
    api_key: str,
    model_name: str,
    json_path: Path,
    existing_mode: str,
    retry_attempts: int,
    retry_wait_seconds: float,
    http_timeout_seconds: float,
) -> bool:
    payload = load_json(json_path)
    if should_skip_file(payload, existing_mode):
        return False

    false_class, detection_items = collect_questions_for_detection(payload, existing_mode)
    if not detection_items:
        return False

    file_id = str(payload.get("metadata", {}).get("id", json_path.stem))
    labels_by_id = retry_call(
        lambda: call_detector(
            api_key=api_key,
            model_name=model_name,
            file_id=file_id,
            false_class=false_class,
            questions=detection_items,
            timeout_seconds=http_timeout_seconds,
        ),
        attempts=retry_attempts,
        wait_seconds=retry_wait_seconds,
        context=f"file={file_id}",
        retryable=is_retryable_api_error,
    )
    updated_payload = update_payload_with_labels(payload, labels_by_id, existing_mode)
    write_json(json_path, updated_payload)
    return True


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    output_dir = args.output_dir
    only_ids = set(args.only_id) if args.only_id else None

    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}")
        print("Summary: folders=0, found=0, updated=0, skipped=0, failed=0")
        raise SystemExit(1)

    api_key = get_api_key()

    output_subdirs = iter_output_subdirs(output_dir, args.output_subdir)
    total_found = 0
    total_updated = 0
    total_skipped = 0
    total_failed = 0
    valid_folder_count = 0

    for output_subdir in output_subdirs:
        if not output_subdir.exists():
            print(f"Warning: output subdirectory not found: {output_subdir}")
            continue
        if not output_subdir.is_dir():
            print(f"Warning: output subdirectory is not a directory: {output_subdir}")
            continue

        valid_folder_count += 1
        json_files = iter_json_files(output_subdir, args.limit, only_ids)
        folder_updated = 0
        folder_skipped = 0
        folder_failed = 0

        def process_json_path(json_path: Path) -> bool:
            return detect_single_file(
                api_key=api_key,
                model_name=args.model_name,
                json_path=json_path,
                existing_mode=args.existing_detect,
                retry_attempts=args.retry_attempts,
                retry_wait_seconds=args.retry_wait_seconds,
                http_timeout_seconds=args.http_timeout_seconds,
            )

        if args.num_workers == 1:
            for json_path in json_files:
                try:
                    changed = process_json_path(json_path)
                    if changed:
                        folder_updated += 1
                    else:
                        folder_skipped += 1
                except Exception as exc:
                    folder_failed += 1
                    print(f"Failed: {output_subdir.name}/{json_path.name}: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {
                    executor.submit(process_json_path, json_path): json_path
                    for json_path in json_files
                }
                for future in as_completed(futures):
                    json_path = futures[future]
                    try:
                        changed = future.result()
                        if changed:
                            folder_updated += 1
                        else:
                            folder_skipped += 1
                    except Exception as exc:
                        folder_failed += 1
                        print(f"Failed: {output_subdir.name}/{json_path.name}: {exc}")

        total_found += len(json_files)
        total_updated += folder_updated
        total_skipped += folder_skipped
        total_failed += folder_failed

        print(
            f"Folder {output_subdir.name}: found={len(json_files)}, "
            f"updated={folder_updated}, skipped={folder_skipped}, failed={folder_failed}"
        )

    print(
        f"Summary: folders={valid_folder_count}, found={total_found}, "
        f"updated={total_updated}, skipped={total_skipped}, failed={total_failed}"
    )
    if total_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
