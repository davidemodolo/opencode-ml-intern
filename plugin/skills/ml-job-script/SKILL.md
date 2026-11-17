---
name: ml-job-script
description: Template and rules for scripts submitted with submit_job (PEP 723 dependencies, fail-fast checks, outputs, logging)
---
# Job scripts for submit_job

submit_job runs the script with `uv run --script` in its own job directory. The script must be self-contained. Pass its source or the absolute path of the `.py` file you wrote: the file is copied at submit time.

## Template

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==<latest>",
#   "transformers==<latest>",
#   "datasets==<latest>",
# ]
# ///
import os
import sys
from importlib.metadata import version

import torch

for package in ["torch", "transformers", "datasets"]:
    print(f"{package}=={version(package)}", flush=True)
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))  # set MAX_STEPS in env for a smoke test
# ... load and validate data, train, save to ./outputs
```

## Rules

- Declare every third-party import in the PEP 723 block, pinned with `==` to the latest version you verified.
- Print package versions and CUDA availability first, so a broken environment fails in seconds.
- Validate inputs before expensive work: required dataset columns, model id, no placeholder values in hub_model_id.
- Write every artifact under `./outputs` (relative to the job directory). Push to the Hub only with a real `hub_model_id`.
- Logging: `logging_strategy="steps"`, `logging_first_step=True`, `disable_tqdm=True`, so the job log shows loss lines.
- Read hyperparameters that change between smoke test and full run from environment variables (pass them in submit_job `env`), so the full run uses the exact tested script.
- Smoke test first: submit with `env={"MAX_STEPS": "10"}` and a short timeout, check the result, then submit the full run.
- Set `vram_gb` from the memory estimate and `timeout_minutes` from the expected duration plus a margin.
