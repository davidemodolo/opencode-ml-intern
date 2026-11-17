---
name: lora-sft
description: Supervised fine-tuning with TRL SFTTrainer and a PEFT LoRA adapter, sized for small GPUs (tested on 8 GB)
---
# LoRA SFT with TRL

Run with MAX_STEPS=10 as a smoke test. Verified with trl 1.18.0, transformers 5.23.0, peft 0.21.3 on an 8 GB GPU (Qwen/Qwen2.5-0.5B-Instruct, trl-lib/Capybara). Check the current TRL docs (explore_hf_docs "trl", sft_trainer page) before changing arguments.

## Script

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.15.0",
#   "transformers==5.23.0",
#   "trl==1.18.0",
#   "peft==0.21.3",
#   "datasets==5.2.0",
#   "accelerate==1.16.0",
# ]
# ///
import os
import sys
from importlib.metadata import version

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET = os.environ.get("DATASET", "trl-lib/Capybara")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID")

for package in ["torch", "transformers", "trl", "peft", "datasets", "accelerate"]:
    print(f"{package}=={version(package)}", flush=True)
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

train = load_dataset(DATASET, split="train")
if "messages" not in train.column_names:
    sys.exit(f"dataset {DATASET} has no 'messages' column: {train.column_names}")
if MAX_STEPS > 0:
    train = train.select(range(min(len(train), MAX_STEPS * 8)))

config = SFTConfig(
    output_dir="outputs",
    max_length=1024,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    activation_checkpointing=True,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    logging_strategy="steps",
    logging_steps=1,
    logging_first_step=True,
    disable_tqdm=True,
    save_strategy="no",
    report_to="none",
    push_to_hub=HUB_MODEL_ID is not None,
    hub_model_id=HUB_MODEL_ID,
)
peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")

trainer = SFTTrainer(model=MODEL, args=config, train_dataset=train, peft_config=peft_config)
trainer.train()
trainer.save_model("outputs/adapter")
print("saved adapter to outputs/adapter", flush=True)
if HUB_MODEL_ID:
    trainer.push_to_hub()
    print(f"pushed to https://huggingface.co/{HUB_MODEL_ID}", flush=True)
```

## Notes

- Dataset formats accepted by SFTTrainer: "messages" (conversational), "text", or "prompt"/"completion". Check with hf_inspect_dataset.
- Memory: LoRA on a model of N billion parameters in bf16 needs about 2N GB for the weights plus activations. With 8 GB, models up to about 1.5B fit with batch size 1-2, gradient accumulation and activation checkpointing; use QLoRA (skill qlora) for larger models.
- On OOM keep the effective batch size: halve per_device_train_batch_size and double gradient_accumulation_steps.
- Load the adapter for inference with `peft.AutoPeftModelForCausalLM.from_pretrained("outputs/adapter")`.
