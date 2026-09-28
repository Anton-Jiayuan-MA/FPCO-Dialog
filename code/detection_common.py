"""Shared detector input preparation and label updates."""

import copy
import json
from typing import Any

QUESTION_FIELD = "question"
BASE_FIELD = "base"


def should_skip_file(payload: dict[str, Any], existing_mode: str, *, detect_field: str) -> bool:
    if existing_mode != "skip":
        return False

    questions = payload.get(QUESTION_FIELD)
    if not isinstance(questions, list) or not questions:
        return False

    return all(isinstance(item, dict) and isinstance(item.get(detect_field), bool) for item in questions)


def collect_questions_for_detection(
    payload: dict[str, Any],
    existing_mode: str,
    *,
    detect_field: str,
) -> tuple[str, list[dict[str, Any]]]:
    base = payload.get(BASE_FIELD)
    questions = payload.get(QUESTION_FIELD)
    if not isinstance(base, dict):
        raise ValueError("Missing or invalid 'base' object.")
    if not isinstance(questions, list):
        raise ValueError("Missing or invalid 'question' list.")

    false_class = base.get("false_class")
    if not isinstance(false_class, str) or not false_class.strip():
        raise ValueError("Missing or invalid base.false_class.")

    detection_items: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Each question item must be a JSON object.")

        if existing_mode == "skip" and isinstance(item.get(detect_field), bool):
            continue

        question_id = item.get("id")
        false_premise = item.get("false_premise")
        content = item.get("content")
        modified = item.get("modified")
        response = item.get("response")

        if not isinstance(question_id, int):
            raise ValueError("Question id must be an integer.")
        if not isinstance(false_premise, bool):
            raise ValueError(f"Question {question_id} has invalid false_premise.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Question {question_id} has invalid content.")
        if not isinstance(modified, str) or not modified.strip():
            raise ValueError(f"Question {question_id} has invalid modified question.")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"Question {question_id} has invalid response.")

        detection_items.append(
            {
                "id": question_id,
                "false_premise": false_premise,
                "content": content.strip(),
                "modified": modified.strip(),
                "response": response.strip(),
            }
        )

    return false_class.strip(), detection_items


def build_detection_prompt(file_id: str, false_class: str, questions: list[dict[str, Any]]) -> str:
    questions_text = json.dumps(questions, ensure_ascii=False, indent=2)
    return f"""
You are labeling model responses for the FPCO-Dialog benchmark.

Use only:
- the shared false_class for this file
- false_premise
- content
- modified
- response

Decide whether the response should be labeled detect = true or detect = false.

Decision rule:

1. If false_premise = true:
- detect = true if the response explicitly says the premise is wrong.
- detect = true if the response says the described object, person, attribute, or location is not present or not correct.
- detect = true if the response directly corrects the user, such as saying it is not X but Y.
- detect = true if the response clearly uses the correct premise from content instead of following the false premise in modified.
- detect = false only when the response goes along with the false premise in modified and does not show correction.

2. If false_premise = false:
- detect = false for a normal answer to the correct question.
- detect = true only if the response unnecessarily questions, corrects, or rejects a question that was actually correct.

Shared false_class for this file:
{false_class}

How to use false_class:
- identity: modified changed who or what the target is
- attribute: modified changed one property of the target
- location: modified changed where the target is or its relative position

Use false_class only to understand what kind of false premise was injected.
Do not use any image.
Do not use any outside context.
Return labels for every provided id.
Return JSON only.

File id: {file_id}
Items:
{questions_text}
""".strip()


def build_updated_question_item(source_item: dict[str, Any], detect_value: bool | None, *, detect_field: str) -> dict[str, Any]:
    ordered: dict[str, Any] = {}

    for key in (
        "id",
        "content",
        "modified",
        "false_premise",
        "response",
        "gpt54_detect",
        "gemini31pro_detect",
    ):
        if key == detect_field and detect_value is not None:
            ordered[key] = detect_value
        elif key in source_item:
            ordered[key] = source_item[key]

    if detect_field not in ordered and detect_value is not None:
        ordered[detect_field] = detect_value

    return ordered


def update_payload_with_labels(
    payload: dict[str, Any],
    labels_by_id: dict[int, bool],
    existing_mode: str,
    *,
    detect_field: str,
) -> dict[str, Any]:
    updated = copy.deepcopy(payload)
    questions = updated.get(QUESTION_FIELD)
    if not isinstance(questions, list):
        raise ValueError("Missing or invalid 'question' list.")

    rebuilt_questions: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Each question item must be a JSON object.")

        question_id = item.get("id")
        if not isinstance(question_id, int):
            raise ValueError("Question id must be an integer.")

        detect_value: bool | None = None
        if question_id in labels_by_id:
            detect_value = labels_by_id[question_id]
        elif existing_mode == "skip" and isinstance(item.get(detect_field), bool):
            detect_value = item[detect_field]

        rebuilt_questions.append(build_updated_question_item(item, detect_value, detect_field=detect_field))

    updated[QUESTION_FIELD] = rebuilt_questions
    return updated
