from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from common import DEFAULT_OUTPUT_DIR, DEFAULT_RESULT_DIR, iter_output_subdirs, load_json


BASE_FIELD = "base"
QUESTION_FIELD = "question"
MAX_K = 10
GPT54_FIELD = "gpt54_detect"
GEMINI_FIELD = "gemini31pro_detect"
FALSE_CLASSES = ("identity", "attribute", "location")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cumulative CorrectionTP@K and CorrectionFP@K for GPT-5.4 detect, Gemini 3.1 Pro detect, and their arithmetic average."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help=f"Result directory for CSV files. Default: {DEFAULT_RESULT_DIR}",
    )
    parser.add_argument(
        "--output-subdir",
        nargs="*",
        default=None,
        help="Only process selected model output subdirectories.",
    )
    return parser.parse_args()


def iter_json_files(output_subdir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_subdir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )


def ratio_string(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    return f"{numerator / denominator:.3f}"


def average_rate_string(left: str, right: str) -> str:
    if not left or not right:
        return ""
    return f"{(float(left) + float(right)) / 2:.3f}"


def validate_question_item(item: dict[str, Any], json_path: Path) -> None:
    question_id = item.get("id")
    false_premise = item.get("false_premise")
    gpt54_detect = item.get(GPT54_FIELD)
    gemini_detect = item.get(GEMINI_FIELD)

    if not isinstance(question_id, int):
        raise ValueError(f"{json_path.name}: question id must be an integer.")
    if not isinstance(false_premise, bool):
        raise ValueError(f"{json_path.name}: question {question_id} has invalid false_premise.")
    if not isinstance(gpt54_detect, bool):
        raise ValueError(f"{json_path.name}: question {question_id} is missing valid {GPT54_FIELD}.")
    if not isinstance(gemini_detect, bool):
        raise ValueError(f"{json_path.name}: question {question_id} is missing valid {GEMINI_FIELD}.")


def collect_questions(output_subdir: Path) -> list[dict[str, Any]]:
    all_questions: list[dict[str, Any]] = []

    for json_path in iter_json_files(output_subdir):
        payload = load_json(json_path)
        base = payload.get(BASE_FIELD)
        questions = payload.get(QUESTION_FIELD)
        if not isinstance(base, dict):
            raise ValueError(f"{json_path.name}: missing or invalid 'base' object.")
        if not isinstance(questions, list):
            raise ValueError(f"{json_path.name}: missing or invalid 'question' list.")

        false_class = base.get("false_class")
        if not isinstance(false_class, str) or false_class not in FALSE_CLASSES:
            raise ValueError(f"{json_path.name}: missing or invalid base.false_class.")

        for item in questions:
            if not isinstance(item, dict):
                raise ValueError(f"{json_path.name}: each question item must be a JSON object.")
            validate_question_item(item, json_path)
            all_questions.append(
                {
                    "id": item["id"],
                    "false_premise": item["false_premise"],
                    "false_class": false_class,
                    GPT54_FIELD: item[GPT54_FIELD],
                    GEMINI_FIELD: item[GEMINI_FIELD],
                }
            )

    return all_questions


def compute_metric(candidates: list[dict[str, Any]], field_name: str) -> str:
    # collect_questions has already validated both detector labels.
    return ratio_string(sum(item[field_name] for item in candidates), len(candidates))


def compute_rates(questions: list[dict[str, Any]], k: int) -> dict[str, str]:
    questions_up_to_k = [item for item in questions if item["id"] <= k]
    tp_candidates = [item for item in questions_up_to_k if item["false_premise"]]
    fp_candidates = [item for item in questions_up_to_k if not item["false_premise"]]

    gpt_tp_rate = compute_metric(tp_candidates, GPT54_FIELD)
    gem_tp_rate = compute_metric(tp_candidates, GEMINI_FIELD)
    gpt_fp_rate = compute_metric(fp_candidates, GPT54_FIELD)
    gem_fp_rate = compute_metric(fp_candidates, GEMINI_FIELD)

    return {
        "tp_gpt54": gpt_tp_rate,
        "tp_gemini31pro": gem_tp_rate,
        "tp_average": average_rate_string(gpt_tp_rate, gem_tp_rate),
        "fp_gpt54": gpt_fp_rate,
        "fp_gemini31pro": gem_fp_rate,
        "fp_average": average_rate_string(gpt_fp_rate, gem_fp_rate),
    }


def build_rows(all_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    grouped_questions = {
        "overall": all_questions,
        **{
            false_class: [
                item for item in all_questions
                if item["false_class"] == false_class
            ]
            for false_class in FALSE_CLASSES
        },
    }

    for k in range(1, MAX_K + 1):
        group_rates = {
            group_name: compute_rates(questions, k)
            for group_name, questions in grouped_questions.items()
        }

        row: dict[str, Any] = {"k": k}

        # Place true-positive rates before false-positive rates.
        for group_name in ("overall", *FALSE_CLASSES):
            rates = group_rates[group_name]
            row[f"{group_name}_correctiontpatk_gpt54"] = rates["tp_gpt54"]
            row[f"{group_name}_correctiontpatk_gemini31pro"] = rates["tp_gemini31pro"]
            row[f"{group_name}_correctiontpatk_average"] = rates["tp_average"]

        for group_name in ("overall", *FALSE_CLASSES):
            rates = group_rates[group_name]
            row[f"{group_name}_correctionfpatk_gpt54"] = rates["fp_gpt54"]
            row[f"{group_name}_correctionfpatk_gemini31pro"] = rates["fp_gemini31pro"]
            row[f"{group_name}_correctionfpatk_average"] = rates["fp_average"]

        rows.append(row)

    return rows


def write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metric_suffixes = ("gpt54", "gemini31pro", "average")
    fieldnames = ["k"]
    fieldnames.extend(
        f"{group_name}_correctiontpatk_{suffix}"
        for group_name in ("overall", *FALSE_CLASSES)
        for suffix in metric_suffixes
    )
    fieldnames.extend(
        f"{group_name}_correctionfpatk_{suffix}"
        for group_name in ("overall", *FALSE_CLASSES)
        for suffix in metric_suffixes
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_single_model(output_subdir: Path, result_dir: Path) -> Path:
    all_questions = collect_questions(output_subdir)
    rows = build_rows(all_questions)
    csv_path = result_dir / f"{output_subdir.name}.csv"
    write_csv(csv_path, rows)
    return csv_path


def main() -> None:
    args = parse_args()

    if not args.output_dir.exists():
        print(f"Error: output directory not found: {args.output_dir}")
        print("Summary: models=0, success=0, failed=0")
        raise SystemExit(1)

    output_subdirs = iter_output_subdirs(args.output_dir, args.output_subdir)
    success_count = 0
    failed_count = 0

    for output_subdir in output_subdirs:
        if not output_subdir.exists():
            failed_count += 1
            print(f"Failed: missing output subdirectory: {output_subdir}")
            continue
        if not output_subdir.is_dir():
            failed_count += 1
            print(f"Failed: not a directory: {output_subdir}")
            continue

        try:
            csv_path = process_single_model(output_subdir, args.result_dir)
            success_count += 1
            print(f"Wrote: {csv_path}")
        except Exception as exc:
            failed_count += 1
            print(f"Failed: {output_subdir.name}: {exc}")

    print(
        f"Summary: models={len(output_subdirs)}, success={success_count}, failed={failed_count}"
    )
    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
