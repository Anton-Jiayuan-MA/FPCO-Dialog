"""End-to-end file flow with mock runners, never real API calls or GPU loads."""

import argparse
import ast
import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import benchmark_common
import common
import statistic
from test_code import HAS_OPENAI, sample


class PipelineTests(unittest.TestCase):
    @unittest.skipUnless(HAS_OPENAI, "OpenAI SDK not installed")
    def test_inference_both_judges_and_csv(self):
        import benchmark_api_gpt as benchmark
        import detect_coop_corr_gemini31pro as gemini
        import detect_coop_corr_gpt54 as gpt

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            metadata = root / "sample01.json"
            output = root / "output/model/sample01.json"
            common.write_json(metadata, sample())
            (root / "sample01.jpg").touch()
            args = argparse.Namespace(existing_output="skip", question_field="auto", system_prompt="system", retry_attempts=2, retry_wait_seconds=0)
            runner = SimpleNamespace(generate=Mock(return_value="answer"))
            benchmark.process_file(metadata, root, output, runner, args)
            labels = {index: index >= 4 for index in range(1, 11)}
            with patch.object(gpt, "call_detector", return_value=labels):
                gpt.detect_single_file(Mock(), output, "skip", 2, 0)
            with patch.object(gemini, "call_detector", return_value=labels):
                gemini.detect_single_file("dummy", "model", output, "skip", 2, 0, 5)
            result = statistic.process_single_model(output.parent, root / "results")
            with result.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[0]["overall_correctiontpatk_average"], "")
            self.assertEqual(rows[-1]["overall_correctiontpatk_average"], "1.000")
            self.assertEqual(rows[-1]["overall_correctionfpatk_average"], "0.000")
            self.assertEqual(runner.generate.call_count, 10)

    def test_local_failures_are_not_retried_and_progress_is_saved(self):
        # Compile only the I/O function so this contract can be tested without torch.
        for filename in ("benchmark_local_qwen.py", "benchmark_local_llava.py", "benchmark_local_intern.py"):
            tree = ast.parse((ROOT / "code" / filename).read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_file")
            module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
            generate = Mock(side_effect=RuntimeError("CUDA out of memory"))
            namespace = {
                "load_json": common.load_json, "write_json": common.write_json,
                "init_output_payload": benchmark_common.init_output_payload,
                "build_history": benchmark_common.build_history,
                "generate_response": generate,
            }
            exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                root = Path(folder)
                metadata, output = root / "sample01.json", root / "output.json"
                common.write_json(metadata, sample())
                (root / "sample01.jpg").touch()
                args = argparse.Namespace(existing_output="skip", question_field="auto", system_prompt="system", max_question_id=None)
                runner = SimpleNamespace(is_vision=True, generate=generate)
                with self.assertRaisesRegex(RuntimeError, "out of memory"):
                    namespace["process_file"](metadata, root, output, runner, args)
                generate.assert_called_once()
                self.assertIsNone(common.load_json(output)["question"][0]["response"])

    def test_internvl_auto_dtype_uses_the_loaded_model_dtype(self):
        tree = ast.parse((ROOT / "code/benchmark_local_intern.py").read_text())
        runner_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LocalInternRunner")
        initializer = next(node for node in runner_class.body if getattr(node, "name", None) == "__init__")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), initializer], type_ignores=[])
        model = Mock(dtype=object())
        model.eval.return_value = model
        model.to.return_value = model
        loader = SimpleNamespace(from_pretrained=Mock(return_value=model))
        namespace = {
            "transformers_version": "4.37.2", "resolve_dtype": lambda name: name,
            "should_use_flash_attn": lambda mode: False,
            "AutoModel": loader, "AutoTokenizer": SimpleNamespace(from_pretrained=Mock()),
        }
        exec(compile(ast.fix_missing_locations(module), "internvl_initializer", "exec"), namespace)
        args = SimpleNamespace(model_path=Path("model"), device="cpu", device_map="none", torch_dtype="auto", max_new_tokens=8, temperature=0, top_p=1, system_prompt="system", input_size=448, max_num_tiles=1, use_flash_attn="false")
        runner = SimpleNamespace(_resolve_device_map=lambda mode: None)
        namespace["__init__"](runner, args)
        self.assertIs(runner.torch_dtype, model.dtype)
        self.assertEqual(loader.from_pretrained.call_args.kwargs["torch_dtype"], "auto")

    @unittest.skipUnless(HAS_OPENAI, "OpenAI SDK not installed")
    def test_outer_retry_can_disable_sdk_retries(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "dummy"}), patch("openai.OpenAI") as client:
            common.build_gpt_client("openai", sdk_retries=0)
        client.assert_called_once_with(api_key="dummy", max_retries=0)


if __name__ == "__main__":
    unittest.main()
