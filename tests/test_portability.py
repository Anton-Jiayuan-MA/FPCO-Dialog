"""Portable paths, secure requests, and command-line checks without paid APIs."""

import importlib.util
import io
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import check_dataset
import common
from test_code import HAS_OPENAI, sample


class PathsAndDatasetTests(unittest.TestCase):
    def test_published_layout_precedes_legacy(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            root = Path(folder)
            for name in ("dataset/image", "dataset/metadata_and_questions", "image", "image_metadata"):
                (root / name).mkdir(parents=True)
            self.assertEqual(common.dataset_paths(root), (root / "dataset/image", root / "dataset/metadata_and_questions"))

    def test_legacy_and_missing_layouts(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            root = Path(folder)
            self.assertEqual(common.dataset_paths(root), (root / "dataset/image", root / "dataset/metadata_and_questions"))
            (root / "image").mkdir()
            (root / "image_metadata").mkdir()
            self.assertEqual(common.dataset_paths(root), (root / "image", root / "image_metadata"))

    def test_explicit_environment_overrides(self):
        with patch.dict(os.environ, {"FPCO_DIALOG_IMAGE_DIR": "custom-images", "FPCO_DIALOG_METADATA_DIR": "custom-json"}):
            self.assertEqual(common.dataset_paths(ROOT), (Path("custom-images"), Path("custom-json")))

    def test_published_dataset_and_arbitrary_working_directory(self):
        env = dict(os.environ)
        for name in ("FPCO_DIALOG_BASE_DIR", "FPCO_DIALOG_IMAGE_DIR", "FPCO_DIALOG_METADATA_DIR"):
            env.pop(name, None)
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "code/check_dataset.py"), "--expected-count", "1080"],
                cwd=folder, env=env, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("10800 valid questions", result.stdout)

    def test_invalid_protocol_is_rejected(self):
        payload = sample()
        payload["base"].update(target_description="target", modified_target_description="other")
        check_dataset.validate_sample(payload, "sample01")
        payload["question"][0]["id"] = True
        with self.assertRaises(ValueError):
            check_dataset.validate_sample(payload, "sample01")

    def test_orphaned_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "image").mkdir()
            (root / "metadata").mkdir()
            (root / "image/unpaired.jpg").touch()
            with self.assertRaises(ValueError):
                check_dataset.validate_dataset(root / "image", root / "metadata")


class SecureGeminiTests(unittest.TestCase):
    def test_key_is_in_header_and_certificate_checks_are_enabled(self):
        response = io.BytesIO(b'{"candidates": []}')
        with patch.object(common.urllib.request, "urlopen", return_value=response) as request:
            result = common.gemini_generate_content("dummy-token", "test-model", {"contents": []}, 3)
        self.assertEqual(result, {"candidates": []})
        sent = request.call_args.args[0]
        headers = {name.lower(): value for name, value in sent.header_items()}
        self.assertEqual(headers["x-goog-api-key"], "dummy-token")
        self.assertNotIn("dummy-token", sent.full_url)
        context = request.call_args.kwargs["context"]
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_http_error_is_not_hidden(self):
        error = urllib.error.HTTPError("https://example.invalid", 401, "Unauthorized", None, None)
        with patch.object(common.urllib.request, "urlopen", side_effect=error), self.assertRaises(urllib.error.HTTPError) as caught:
            common.gemini_generate_content("dummy", "model", {}, 1)
        self.assertIs(caught.exception, error)
        self.assertFalse(common.is_retryable_api_error(error))

    def test_nonobject_response_is_rejected(self):
        with patch.object(common.urllib.request, "urlopen", return_value=io.BytesIO(b'[]')), self.assertRaises(common.InvalidModelResponse):
            common.gemini_generate_content("dummy", "model", {}, 1)

    @unittest.skipUnless(shutil.which("openssl"), "openssl required for loopback TLS test")
    def test_untrusted_tls_fails_and_explicit_ca_succeeds(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as folder:
            cert, key = Path(folder) / "cert.pem", Path(folder) / "key.pem"
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost",
                "-keyout", str(key), "-out", str(cert),
            ], check=True, capture_output=True)
            server = HTTPServer(("127.0.0.1", 0), Handler)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                endpoint = f"https://localhost:{server.server_port}/models"
                with patch.object(common, "GEMINI_API_BASE", endpoint), patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(urllib.error.URLError) as caught:
                        common.gemini_generate_content("dummy", "model", {}, 3)
                    self.assertIsInstance(caught.exception.reason, ssl.SSLCertVerificationError)
                    self.assertFalse(common.is_retryable_api_error(caught.exception))
                    with patch.dict(os.environ, {"SSL_CERT_FILE": str(cert)}):
                        self.assertEqual(common.gemini_generate_content("dummy", "model", {}, 3), {"ok": True})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


class CLITests(unittest.TestCase):
    def test_stdlib_commands_have_safe_help(self):
        for name in ("benchmark_api_gemini.py", "detect_coop_corr_gemini31pro.py", "statistic.py", "check_dataset.py"):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "code" / name), "--help"], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

    @unittest.skipUnless(HAS_OPENAI, "OpenAI SDK not installed")
    def test_sdk_commands_have_safe_help(self):
        for name in ("benchmark_api_gpt.py", "benchmark_api_qwen.py", "detect_coop_corr_gpt54.py", "generate_right_question_gpt54.py", "question_error_inject_gpt54.py"):
            result = subprocess.run([sys.executable, "-B", str(ROOT / "code" / name), "--help"], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

    @unittest.skipUnless(HAS_OPENAI, "OpenAI SDK not installed")
    def test_dataset_construction_requires_explicit_overwrite(self):
        import generate_right_question_gpt54 as generation
        import question_error_inject_gpt54 as injection

        for module in (generation, injection):
            with patch.object(sys, "argv", [module.__file__]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                module.parse_args()
            self.assertEqual(caught.exception.code, 2)

    def test_no_machine_specific_paths_or_insecure_tls_in_sources(self):
        for path in (ROOT / "code").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("/data3/", source, path.name)
            self.assertNotIn("_create_unverified_context", source, path.name)
            self.assertNotIn("sys.path.insert", source, path.name)


if __name__ == "__main__":
    unittest.main()
