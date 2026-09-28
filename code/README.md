# Installation and evaluation

Run the commands below from the repository root. Python 3.10 is the reference
version. On Debian/Ubuntu, install the matching `python3.10-venv` package if
creating a virtual environment reports that `ensurepip` is missing. Alternatively,
create a Python 3.10 Conda environment. On Windows, activate a venv with
`.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

## 1. Install only what you need

| Workflow | Dependency file | Notes |
| --- | --- | --- |
| API benchmarks and detectors | `requirements.txt` | No GPU or model weights needed |
| Local Qwen / LLaVA | `requirements-local.txt` | Transformers 5.4.0 |
| Local InternVL2.5 | `requirements-internvl.txt` | Separate environment, Transformers 4.37.2 |
| Dataset checks / statistics / Gemini HTTP calls | Standard library | No third-party dependencies required |

Start with the API environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The files pin the principal runtime dependencies from the reference environment;
they are not full cross-platform lockfiles. Do not install the two local-model
dependency files into the same environment. There is no private dependency
directory or machine-specific import path.

## 2. Validate the published dataset

```bash
python code/check_dataset.py --expected-count 1080
```

This read-only check verifies image/JSON pairing, question IDs and text, the
10-turn protocol, and false-premise flags. It makes no API calls. It does not
decode image pixels or assess the semantic quality of the questions.

Defaults resolve relative to the project directory, not the shell's current
directory:

| Content | Default location |
| --- | --- |
| Images | `dataset/image/` |
| Metadata and questions | `dataset/metadata_and_questions/` |
| Generated model responses and detector labels | `output/<model>/` |
| Published summary CSVs | `result/` |

The original `image/` and `image_metadata/` layout remains supported when the
corresponding published directory is absent. CLI `--image-dir`, `--metadata-dir`,
and `--output-dir` override the defaults. `FPCO_DIALOG_BASE_DIR`,
`FPCO_DIALOG_IMAGE_DIR`, and `FPCO_DIALOG_METADATA_DIR` also provide environment
overrides. Use absolute paths when invoking scripts from another directory.

## 3. Configure credentials

| Provider | Environment variable |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Qwen / DashScope | `QWEN_API_KEY` or `DASHSCOPE_API_KEY` |

Use your platform's secret settings or export variables in the shell. In Bash,
this reads a key without echoing it or putting its value in command history:

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
```

For another provider, replace the variable name. `GPT_API_PROVIDER` selects
`openai` (default) or `openrouter`; the GPT benchmark also accepts
`--api-provider openrouter`. Scripts do not load `.env` files automatically.
Never commit credentials or paste them into error reports.

Both Gemini scripts verify HTTPS certificates and send credentials in the
`x-goog-api-key` header, as shown in the [Gemini authentication guide](https://ai.google.dev/gemini-api/docs/api-key).
For a trusted corporate proxy, set `SSL_CERT_FILE` to your administrator-provided
CA bundle (or configure `SSL_CERT_DIR`). Do not disable verification. A missing
or incorrect CA bundle must be repaired before retrying requests.

## 4. Run a small benchmark first

Each image has 10 sequential turns. `--limit 1` selects one image, not one API
call. These examples incur provider charges and require access to the selected
model. Defaults preserve the experiment model names; use `--model-name` if you
need a different available model, and record that change in your results.

```bash
python code/benchmark_api_gpt.py --model-name gpt-4o --output-subdir smoke_gpt4o --limit 1
python code/benchmark_api_gemini.py --model-name gemini-2.5-flash --output-subdir smoke_gemini --limit 1
python code/benchmark_api_qwen.py --model-name qwen-vl-max --output-subdir smoke_qwen --limit 1
```

Run only the provider you intend to use. After the smoke run succeeds, remove
`--limit 1` for the full dataset. API benchmarks accept `--num-workers`; begin
with the default of one to avoid rate limits. The Qwen API runner uses the
DashScope mainland-China compatible endpoint, so its key must match that service.

Responses are saved after each turn. Re-running the same command resumes completed
responses by default; `--existing-output overwrite` regenerates them. Use a new
`--output-subdir` when changing a model, prompt, decoding settings, or question
set: resume matches question IDs and does not establish that two configurations
are scientifically equivalent.

## 5. Detect correction behavior, then calculate metrics

Run both detectors on the same benchmark output before running statistics:

```bash
python code/detect_coop_corr_gpt54.py --output-subdir smoke_gpt4o
python code/detect_coop_corr_gemini31pro.py --output-subdir smoke_gpt4o
python code/statistic.py --output-subdir smoke_gpt4o --result-dir runs/results
```

The detectors require their respective API keys and write labels into
`output/smoke_gpt4o/*.json`. Both default to skipping existing labels. Use
`--existing-detect overwrite` only when intentionally rerunning detection. Both
support `--model-name`, but changing a judge changes the evaluation; the legacy
field names `gpt54_detect` and `gemini31pro_detect` do not change automatically.

Statistics require both boolean labels for every included question. TP metrics
are blank where no false-premise questions have occurred yet, rather than being
reported as zero. The `runs/results` example keeps your new CSVs separate from
the published `result/` files. Raw per-model output JSONs must be generated first;
the published CSVs alone are not enough to recompute the metrics.

## 6. Optional local-model inference

Model weights are not included. Downloads can be several GB and inference needs
sufficient RAM/VRAM. Only load checkpoints you trust: these runners allow custom
model code through `trust_remote_code=True`.

Create a separate environment for Qwen/LLaVA. Install a matching PyTorch wheel
for your hardware first. The following is the reference CUDA 12.8 example; use
the [official PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/#v2110)
for CPU or other supported CUDA builds.

```bash
python3.10 -m venv .venv-local
source .venv-local/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-local.txt
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models/Qwen2.5-VL-3B-Instruct
python code/benchmark_local_qwen.py --model-path models/Qwen2.5-VL-3B-Instruct --output-subdir qwen_local --limit 1
```

For LLaVA, download a compatible LLaVA-NeXT or OneVision checkpoint and run
`benchmark_local_llava.py --model-path <checkpoint-directory> --limit 1` in the
same environment. Both runners accept `--device-map auto` for multi-device
loading; CPU runs can use `--device cpu --torch-dtype float32 --device-map none`
but can be very slow and require substantial RAM. Download commands use the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).

InternVL2.5 uses a separate environment for its custom model code. The pinned
version follows the reference experiment setup and the model's
[Transformers-based interface](https://huggingface.co/OpenGVLab/InternVL2_5-8B#quick-start).

```bash
python3.10 -m venv .venv-internvl
source .venv-internvl/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-internvl.txt
hf download OpenGVLab/InternVL2_5-8B --local-dir models/InternVL2_5-8B
python code/benchmark_local_intern.py --model-path models/InternVL2_5-8B --torch-dtype bfloat16 --use-flash-attn false --limit 1
```

FlashAttention is optional and off by default (`auto` also leaves it off).
Only enable it if you have installed a compatible build. CUDA out-of-memory
errors stop the current run immediately: reduce image tiles (`--max-num-tiles`
for InternVL), token budget, or model size, or provide more memory. Completed
turns remain saved. The local runners no longer accept retry flags; retrying the
same memory-heavy operation does not fix its resource requirements.

## 7. Dataset construction is a separate, destructive workflow

The supplied metadata is already ready for benchmarking. Do not run construction
scripts unless you intend to create a new dataset. They overwrite the selected
JSON files, require paid API calls, and now require explicit `--overwrite`.
Work on a copy and keep the published dataset unchanged:

```bash
mkdir -p runs
cp -R dataset/metadata_and_questions runs/new_metadata
python code/generate_right_question_gpt54.py --metadata-dir runs/new_metadata --limit 1 --overwrite
python code/question_error_inject_gpt54.py --metadata-dir runs/new_metadata --limit 1 --overwrite
```

Both support `--model-name`, `--limit`, and `--only-id`; the generation script
also supports `--image-dir`. Question counts, prompt text, and the first-three
correct / last-seven false-premise protocol are unchanged.

## 8. Offline tests and failure handling

```bash
python -B -m unittest discover -s tests -v
```

Tests check credentials, retries, resume/history, label validation, metrics,
dataset paths, CLI help, and HTTPS verification. No test calls a provider or
loads model weights. The TLS test uses a temporary localhost server and requires
OpenSSL and permission to open a loopback socket. SDK-specific tests are skipped
when the OpenAI SDK is absent. GitHub Actions runs the API/offline suite and
dataset check; it does not run paid APIs or GPU inference.

Transient API failures and malformed responses have bounded retries; authentication,
input, certificate, and file-writing errors do not. API benchmark/detector SDK
retries are disabled where an outer retry loop already exists. Batch failures
are reported and return a nonzero exit status; local-model failures stop promptly.
After correcting an error, rerun to resume. A HTTP 401/403 requires checking the
key, provider, or model access; a 404 can indicate an unavailable model name.

The retained checks protect external inputs, saved progress, or metric validity.
Removed compatibility code includes the obsolete LLaVA auto-class fallback,
the no-op statistics flag, redundant I/O exception wrapping, and local OOM retries.
Processor-specific adaptations remain where different checkpoint formats need them.
