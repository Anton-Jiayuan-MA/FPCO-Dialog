"""Offline regression tests; no API requests or model weights are needed."""

import copy
import importlib.util
import io
import json
import os
import ssl
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import benchmark_common as benchmark
import common
import detect_coop_corr_gemini31pro as gemini_detector
import detection_common as detection
import statistic

HAS_OPENAI = importlib.util.find_spec("openai") is not None
if HAS_OPENAI:
    import benchmark_api_gpt as gpt
    import benchmark_api_qwen as qwen
    import detect_coop_corr_gpt54 as gpt_detector
    import question_error_inject_gpt54 as injection
    from openai import APIConnectionError


def sample():
    return {
        "metadata": {"id": "sample01"},
        "base": {"false_class": "identity"},
        "question": [
            {
                "id": index,
                "content": f"Correct question {index}?",
                "modified": f"Modified question {index}?",
                "false_premise": index >= 4,
                "response": f"Answer {index}",
            }
            for index in range(1, 11)
        ],
    }


class CredentialsAndRetryTests(unittest.TestCase):
    def test_credentials_have_no_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            for provider in ("openai", "openrouter"):
                with self.assertRaises(ValueError):
                    common.get_gpt_api_key(provider)
            with self.assertRaises(ValueError):
                gemini_detector.get_api_key()

    def test_credentials_strip_whitespace_and_support_aliases(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "  ", "GOOGLE_API_KEY": " dummy "}, clear=True):
            self.assertEqual(gemini_detector.get_api_key(), "dummy")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "dummy"}, clear=True):
            self.assertEqual(common.require_api_key("QWEN_API_KEY", "DASHSCOPE_API_KEY"), "dummy")

    def test_provider_and_model_names(self):
        with self.assertRaises(ValueError):
            common.get_gpt_api_key("unsupported")
        self.assertEqual(common.resolve_model_name("gpt-4o", "openrouter"), "openai/gpt-4o")
        self.assertEqual(common.resolve_model_name("openai/gpt-4o", "openrouter"), "openai/gpt-4o")
        self.assertEqual(common.resolve_model_name("gpt-4o", "openai"), "gpt-4o")

    def run_retry(self, operation, **kwargs):
        with redirect_stdout(io.StringIO()):
            return common.retry_call(operation, attempts=3, wait_seconds=0, context="test", **kwargs)

    def test_transient_failure_retries_then_succeeds(self):
        operation = Mock(side_effect=[TimeoutError(), "ok"])
        self.assertEqual(self.run_retry(operation, retryable=common.is_retryable_api_error), "ok")
        self.assertEqual(operation.call_count, 2)

    def test_permanent_failures_are_not_retried(self):
        for error in (ValueError("input"), FileNotFoundError("image"), PermissionError("disk")):
            with self.subTest(error=type(error).__name__):
                operation = Mock(side_effect=error)
                with self.assertRaises(type(error)):
                    self.run_retry(operation, retryable=common.is_retryable_api_error)
                operation.assert_called_once()

    def test_http_status_classification(self):
        for code in (400, 401, 403, 404, 408, 409, 429, 500, 503):
            error = urllib.error.HTTPError("https://example.invalid", code, "test", None, None)
            self.assertEqual(common.is_retryable_api_error(error), code in (408, 409, 429, 500, 503))
            wrapped = RuntimeError("request failed")
            wrapped.__cause__ = error
            self.assertEqual(common.is_retryable_api_error(wrapped), common.is_retryable_api_error(error))

    def test_retry_exhaustion_preserves_original_exception(self):
        error = TimeoutError("temporary")
        operation = Mock(side_effect=error)
        with self.assertRaises(TimeoutError) as caught:
            self.run_retry(operation, retryable=common.is_retryable_api_error)
        self.assertIs(caught.exception, error)
        self.assertEqual(operation.call_count, 3)

    def test_invalid_retry_options_do_not_run_operation(self):
        operation = Mock()
        for attempts, wait in ((0, 1), (1, -1)):
            with self.assertRaises(ValueError):
                common.retry_call(operation, attempts=attempts, wait_seconds=wait, context="test", retryable=lambda exc: True)
        operation.assert_not_called()

    def test_interrupts_propagate(self):
        operation = Mock(side_effect=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.run_retry(operation, retryable=lambda exc: True)
        operation.assert_called_once()

    def test_bad_model_output_is_retryable(self):
        self.assertTrue(common.is_retryable_api_error(common.InvalidModelResponse("bad output")))
        self.assertTrue(common.is_retryable_api_error(json.JSONDecodeError("bad", "", 0)))

    def test_certificate_errors_are_not_retried(self):
        error = ssl.SSLCertVerificationError("invalid certificate")
        self.assertFalse(common.is_retryable_api_error(error))
        self.assertFalse(common.is_retryable_api_error(urllib.error.URLError(error)))

    @unittest.skipUnless(HAS_OPENAI, "OpenAI SDK not installed")
    def test_sdk_connection_wrapper_and_certificate_cause(self):
        import httpx

        error = APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        error.__cause__ = httpx.ConnectError("network")
        self.assertTrue(common.is_retryable_api_error(error, (APIConnectionError,)))
        error.__cause__.__cause__ = ssl.SSLCertVerificationError("certificate")
        self.assertFalse(common.is_retryable_api_error(error, (APIConnectionError,)))


class BenchmarkDataTests(unittest.TestCase):
    def test_question_selection(self):
        item = {"modified": " modified ", "content": " original "}
        self.assertEqual(benchmark.pick_question_text(item, "auto"), "modified")
        self.assertEqual(benchmark.pick_question_text(item, "content"), "original")
        self.assertEqual(benchmark.pick_question_text({"content": "original"}, "auto"), "original")
        with self.assertRaises(ValueError):
            benchmark.pick_question_text({}, "modified")

    def test_resume_and_overwrite_do_not_mutate_source(self):
        source = sample()
        saved = copy.deepcopy(source)
        snapshot = copy.deepcopy(source)
        resumed = benchmark.init_output_payload(source, saved, "skip")
        self.assertEqual(resumed, source)
        overwritten = benchmark.init_output_payload(source, saved, "overwrite")
        self.assertTrue(all(item["response"] is None for item in overwritten["question"]))
        self.assertEqual(source, snapshot)

    def test_invalid_metadata_or_questions_fail_at_input_boundary(self):
        for payload in ([], {"question": []}, {"metadata": {}, "question": {}}, {"metadata": {}, "question": [None]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                benchmark.init_output_payload(payload, None, "skip")

    def test_history_requires_prior_responses(self):
        questions = sample()["question"]
        history = benchmark.build_history(questions, 2, "auto", "system")
        self.assertEqual([item["role"] for item in history], ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(history[-1]["content"], "Modified question 3?")
        questions[0]["response"] = None
        with self.assertRaises(ValueError):
            benchmark.build_history(questions, 1, "auto", "system")

    def test_json_roundtrip_and_object_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "sample.json"
            common.write_json(path, sample())
            self.assertEqual(common.load_json(path), sample())
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                common.load_json(path)

    @unittest.skipUnless(HAS_OPENAI, "OpenAI SDK not installed")
    def test_process_file_resume_does_not_repeat_generation(self):
        from argparse import Namespace

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            metadata = root / "sample01.json"
            output = root / "output.json"
            common.write_json(metadata, sample())
            (root / "sample01.jpg").touch()
            runner = SimpleNamespace(generate=Mock(return_value="answer"))
            args = Namespace(existing_output="skip", question_field="auto", system_prompt="system", retry_attempts=2, retry_wait_seconds=0)
            with redirect_stdout(io.StringIO()):
                gpt.process_file(metadata, root, output, runner, args)
                gpt.process_file(metadata, root, output, runner, args)
            self.assertEqual(runner.generate.call_count, 10)
            self.assertTrue(all(item["response"] == "answer" for item in common.load_json(output)["question"]))


class DetectorTests(unittest.TestCase):
    def test_skip_only_completed_detection(self):
        payload = sample()
        self.assertFalse(gemini_detector.should_skip_file(payload, "skip"))
        for item in payload["question"]:
            item["gemini31pro_detect"] = False
        self.assertTrue(gemini_detector.should_skip_file(payload, "skip"))
        self.assertFalse(gemini_detector.should_skip_file(payload, "overwrite"))

    def test_collect_requires_responses_and_boolean_premises(self):
        payload = sample()
        false_class, questions = gemini_detector.collect_questions_for_detection(payload, "skip")
        self.assertEqual(false_class, "identity")
        self.assertEqual(len(questions), 10)
        for key, invalid in (("response", " "), ("false_premise", "false"), ("id", "1")):
            changed = copy.deepcopy(payload)
            changed["question"][0][key] = invalid
            with self.assertRaises(ValueError):
                gemini_detector.collect_questions_for_detection(changed, "skip")

    def test_label_update_preserves_other_detector(self):
        payload = sample()
        payload["question"][0]["gpt54_detect"] = True
        updated = gemini_detector.update_payload_with_labels(payload, {1: False}, "skip")
        self.assertTrue(updated["question"][0]["gpt54_detect"])
        self.assertFalse(updated["question"][0]["gemini31pro_detect"])
        self.assertNotIn("gemini31pro_detect", payload["question"][0])

    def test_gemini_accepts_only_unambiguous_boolean_values(self):
        for value in (True, 1, "true", "yes", "1"):
            self.assertTrue(gemini_detector.coerce_detect_value(value))
        for value in (False, 0, "false", "no", "0"):
            self.assertFalse(gemini_detector.coerce_detect_value(value))
        for value in (2, None, "maybe", [], 0.5):
            with self.assertRaises(ValueError):
                gemini_detector.coerce_detect_value(value)

    def test_gemini_fenced_and_list_json(self):
        labels = [{"id": 1, "detect": True}]
        self.assertEqual(gemini_detector.parse_detector_json("```json\n" + json.dumps(labels) + "\n```"), {"labels": labels})

    def test_gemini_rejects_missing_duplicate_and_nonobject_labels(self):
        cases = ({"labels": []}, {"labels": [{"id": 1, "detect": True}] * 2}, [], {"labels": [None]})
        for invalid in cases:
            response = {"candidates": [{"content": {"parts": [{"text": json.dumps(invalid)}]}}]}
            with patch.object(gemini_detector, "post_generate_content", return_value=response), self.assertRaises(common.InvalidModelResponse):
                gemini_detector.call_detector("dummy", "model", "sample", "identity", sample()["question"][:1], 10)

    def test_write_failure_does_not_repeat_paid_detection(self):
        with patch.object(gemini_detector, "load_json", return_value=sample()), \
             patch.object(gemini_detector, "call_detector", return_value={i: False for i in range(1, 11)}) as call, \
             patch.object(gemini_detector, "write_json", side_effect=PermissionError("disk")):
            with self.assertRaises(PermissionError):
                gemini_detector.detect_single_file("dummy", "model", Path("sample.json"), "skip", 3, 0, 10)
        call.assert_called_once()

    def test_partial_batch_failure_exits_unsuccessfully(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            model = output / "model"
            model.mkdir()
            common.write_json(model / "sample.json", sample())
            for workers in (1, 2):
                args = SimpleNamespace(
                    num_workers=workers, output_dir=output, only_id=None,
                    output_subdir=None, limit=None, model_name="model",
                    existing_detect="skip", retry_attempts=2,
                    retry_wait_seconds=0, http_timeout_seconds=10,
                )
                with self.subTest(workers=workers), \
                     patch.object(gemini_detector, "parse_args", return_value=args), \
                     patch.object(gemini_detector, "get_api_key", return_value="dummy"), \
                     patch.object(gemini_detector, "detect_single_file", side_effect=ValueError("bad sample")), \
                     redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    gemini_detector.main()
                self.assertEqual(caught.exception.code, 1)

    @unittest.skipUnless(HAS_OPENAI, "OpenAI SDK not installed")
    def test_gpt_write_failure_does_not_repeat_paid_detection(self):
        with patch.object(gpt_detector, "load_json", return_value=sample()), \
             patch.object(gpt_detector, "call_detector", return_value={i: False for i in range(1, 11)}) as call, \
             patch.object(gpt_detector, "write_json", side_effect=PermissionError("disk")):
            with self.assertRaises(PermissionError):
                gpt_detector.detect_single_file(Mock(), Path("sample.json"), "skip", 3, 0)
        call.assert_called_once()


class ProtocolAndStatisticsTests(unittest.TestCase):
    @unittest.skipUnless(HAS_OPENAI, "OpenAI SDK not installed")
    def test_injection_protocol_and_category_boundaries(self):
        self.assertEqual(injection.assign_false_premises(sample()["question"]), {i: i >= 4 for i in range(1, 11)})
        for end, expected in ((1, "identity"), (30, "identity"), (31, "attribute"), (60, "attribute"), (61, "location"), (90, "location")):
            self.assertEqual(injection.infer_false_class_from_file_id(f"sample{end:02}"), expected)
        for questions in (sample()["question"][:9], list(reversed(sample()["question"]))):
            with self.assertRaises(ValueError):
                injection.assign_false_premises(questions)

    def test_empty_denominator_stays_blank_not_zero(self):
        self.assertEqual(statistic.ratio_string(0, 0), "")
        self.assertEqual(statistic.compute_metric([], "gpt54_detect"), "")
        self.assertEqual(statistic.average_rate_string("", "0.500"), "")

    def test_metrics_and_column_order(self):
        questions = sample()["question"]
        for item in questions:
            item.update(false_class="identity", gpt54_detect=item["false_premise"], gemini31pro_detect=False)
        rows = statistic.build_rows(questions)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[-1]["overall_correctiontpatk_gpt54"], "1.000")
        self.assertEqual(rows[-1]["overall_correctiontpatk_average"], "0.500")
        self.assertEqual(rows[-1]["overall_correctionfpatk_average"], "0.000")
        self.assertEqual(rows[0]["overall_correctiontpatk_gpt54"], "")
        self.assertEqual(len(rows[0]), 25)


if __name__ == "__main__":
    unittest.main()
