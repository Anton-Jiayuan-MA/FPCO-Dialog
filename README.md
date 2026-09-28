# FPCO-Dialog

Official code and dataset for FPCO-Dialog, accepted at EMNLP2026 Main Conference.

**arXiv:** https://arxiv.org/abs/2609.03331

## Overview

[![Overview of FPCO-Dialog](figure/main.png)](figure/main.pdf)

Click the figure to view or download the original PDF.

## Quick start

Use Python 3.10 for the reference environment. The published dataset already
contains the questions; you do not need to regenerate them.

```bash
git clone https://github.com/lab-klc/FPCO-Dialog.git
cd FPCO-Dialog
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/check_dataset.py --expected-count 1080
```

The check is read-only and needs no API key or GPU. See the
[installation and evaluation guide](code/README.md) for API credentials, local
model environments, inference, response detection, statistics, and troubleshooting.
API calls incur provider charges; local inference requires downloaded model
weights and sufficient hardware. API model availability depends on your account.

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{fpcodialog_emnlp26,
  title={FPCO-Dialog: A Multi-Turn False-Premise Benchmark for Correction and Cooperation in Vision-Language Models},
  author={Jiayuan Ma and Yuqi Lu and Weiyang Guo and Chenrui Wang and Junyi Shu and Xuebo Liu and Min Zhang and Jing Li},
  booktitle={The 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026}
}
```
