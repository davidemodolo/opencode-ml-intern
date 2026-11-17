---
name: qlora
description: 4-bit QLoRA fine-tuning with bitsandbytes to fit 1.5B-8B models on GPUs with 8-16 GB of memory
---
# QLoRA

QLoRA loads the base model in 4-bit NF4 with bitsandbytes and trains a LoRA adapter on top. It changes the precision of the frozen base weights, so use it when the user asked for it or agreed to it.

Start from the lora-sft skill script and change the model loading:

```python
# add to the PEP 723 dependencies: "bitsandbytes==<latest>"
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quantization, dtype=torch.bfloat16)
trainer = SFTTrainer(model=model, args=config, train_dataset=train, peft_config=peft_config)
```

## Memory guide (weights only)

| Parameters | bf16 LoRA | 4-bit QLoRA |
|---|---|---|
| 0.5B | 1 GB | 0.4 GB |
| 1.5B | 3 GB | 1.1 GB |
| 3B | 6 GB | 2 GB |
| 7-8B | 15 GB | 5 GB |

Activations come on top: keep activation checkpointing on, batch size 1-2 and max_length as the task needs.

Check the bitsandbytes and transformers quantization docs (explore_hf_docs "transformers", quantization pages) for the current argument names before submitting.
