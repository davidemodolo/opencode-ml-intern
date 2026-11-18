---
name: multi-gpu
description: Multi-GPU jobs with submit_job (gpus=N on local, multi-GPU hardware on remote backends) using torchrun DDP, FSDP or device_map="auto" from a PEP 723 script
---
# Multi-GPU jobs

On the `local` backend, `submit_job(..., gpus=N)` gives the job N GPUs together: `CUDA_VISIBLE_DEVICES` lists them and `torch.cuda.device_count()` returns N. `vram_gb` is the VRAM the job needs across all of them. On a remote backend, pick a hardware option with several GPUs (`a10g-largex2` on hf_jobs, `A100:2` on skypilot) and leave `gpus` unset.

Pick the approach from what does not fit on one GPU:

| Situation | Approach |
|---|---|
| The model fits on one GPU, training should go faster | DDP: one process per GPU, each with a full copy of the model |
| The model, gradients and optimizer state do not fit on one GPU | FSDP: parameters, gradients and optimizer state sharded across GPUs |
| Inference or evaluation of a model larger than one GPU | `device_map="auto"` in a single process |

## DDP and FSDP: the script launches itself with torchrun

The job runs `uv run --script script.py` once. The script starts one worker per GPU with `torch.distributed.run`, which runs the same file with the same Python environment:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==<latest>",
#   "transformers==<latest>",
#   "trl==<latest>",
#   "datasets==<latest>",
#   "accelerate==<latest>",
# ]
# ///
import os
import sys

import torch

if "LOCAL_RANK" not in os.environ and torch.cuda.device_count() > 1:
    from torch.distributed.run import main as torchrun

    sys.exit(torchrun(["--nproc-per-node", str(torch.cuda.device_count()), sys.argv[0], *sys.argv[1:]]))

rank = int(os.environ.get("RANK", "0"))
if rank == 0:
    print("GPUs:", torch.cuda.device_count(), flush=True)
# ... build the trainer as usual; Trainer and the TRL trainers use DDP under torchrun
```

Rules:

- Batch size: `per_device_train_batch_size` is per GPU, so the effective batch is per-device batch x GPUs x gradient_accumulation_steps. Lower gradient accumulation to keep the effective batch of a single-GPU recipe.
- Print, save and push from rank 0 only (`trainer.is_world_process_zero()` or `RANK == 0`); the Trainer already saves from the main process.
- Leave `device_map` unset on a model trained with DDP or FSDP: each process places its own copy or shard.
- Gradient checkpointing with DDP: set `gradient_checkpointing_kwargs={"use_reentrant": False}`.

## FSDP

Same launcher, plus FSDP in the training arguments. Verify the option names in the current Transformers docs (`fetch_hf_docs` on the FSDP page) before use:

```python
args = SFTConfig(
    ...,
    fsdp="full_shard auto_wrap",
    fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"]},  # the model's decoder layer class
)
```

Find the decoder layer class name in the model's `modeling_*.py` (`hf_repo_files`, or `model.model.layers[0].__class__.__name__` in a quick check).

## device_map="auto" for inference

A single process spreads the layers over all visible GPUs:

```python
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
```

No torchrun. The GPUs run one after the other: a model larger than one GPU fits, at single-GPU speed.

## Smoke test

Run the multi-GPU script with `MAX_STEPS=10` first: a broken process group, a wrong layer class or a rank mismatch fails in the first minute.
