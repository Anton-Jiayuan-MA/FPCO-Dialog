"""Shared benchmark data validation, resume state, and multi-turn history."""

import copy
from typing import Any


def pick_question_text(question_item: dict[str, Any], question_field: str) -> str:
    if question_field == "modified":
        candidate = question_item.get("modified")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        raise ValueError("Question field 'modified' is missing or empty.")
    if question_field == "content":
        candidate = question_item.get("content")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        raise ValueError("Question field 'content' is missing or empty.")

    for field_name in ("modified", "content"):
        candidate = question_item.get(field_name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise ValueError("No usable question text found in 'modified' or 'content'.")

def build_question_output(source_item: dict[str, Any], response: str | None) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in ("id", "content", "modified", "false_premise", "response", "gpt54_detect", "gemini31pro_detect"):
        if key in source_item:
            ordered[key] = source_item[key]
    ordered["response"] = response
    return ordered

def init_output_payload(
    source_payload: dict[str, Any],
    existing_payload: dict[str, Any] | None,
    existing_mode: str,
) -> dict[str, Any]:
    if not isinstance(source_payload, dict):
        raise ValueError("Source payload must be a JSON object.")
    if not isinstance(source_payload.get("metadata"), dict):
        raise ValueError("Source payload is missing a valid 'metadata' object.")
    output_payload = copy.deepcopy(source_payload)
    source_questions = source_payload.get("question")
    if not isinstance(source_questions, list):
        raise ValueError("Source payload is missing a valid 'question' list.")

    existing_questions_by_id: dict[Any, dict[str, Any]] = {}
    if existing_payload and isinstance(existing_payload.get("question"), list):
        existing_questions_by_id = {
            item.get("id"): item
            for item in existing_payload["question"]
            if isinstance(item, dict)
        }

    rebuilt_questions = []
    for item in source_questions:
        if not isinstance(item, dict):
            raise ValueError("Each question item must be a JSON object.")

        response = None
        if existing_mode == "skip":
            existing_item = existing_questions_by_id.get(item.get("id"))
            if isinstance(existing_item, dict):
                existing_response = existing_item.get("response")
                if isinstance(existing_response, str) and existing_response.strip():
                    response = existing_response

        rebuilt_questions.append(build_question_output(item, response))

    output_payload["question"] = rebuilt_questions
    return output_payload

def build_history(
    question_items: list[dict[str, Any]],
    current_index: int,
    question_field: str,
    system_prompt: str,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    for index in range(current_index):
        previous_item = question_items[index]
        previous_question = pick_question_text(previous_item, question_field)
        previous_response = previous_item.get("response")

        history.append({"role": "user", "content": previous_question})
        if not isinstance(previous_response, str) or not previous_response.strip():
            raise ValueError(
                f"Question {previous_item.get('id')} has no saved response, cannot construct multi-turn history."
            )
        history.append({"role": "assistant", "content": previous_response.strip()})

    current_question = pick_question_text(question_items[current_index], question_field)
    history.append({"role": "user", "content": current_question})
    return history
