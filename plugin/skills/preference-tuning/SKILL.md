---
name: preference-tuning
description: DPO and GRPO with TRL, dataset formats, reward functions and memory tips for small GPUs
---
# Preference and RL tuning with TRL

Always confirm the current API with explore_hf_docs "trl" (dpo_trainer and grpo_trainer pages) and a current example from github_find_examples (repo trl).

## DPO

- Dataset columns: "prompt", "chosen", "rejected" (conversational or plain text). Check with hf_inspect_dataset.
- Start from an SFT model or adapter; DPO on a raw base model usually underperforms.
- Shape:

```python
from trl import DPOConfig, DPOTrainer
config = DPOConfig(output_dir="outputs", beta=0.1, learning_rate=5e-6, per_device_train_batch_size=1,
                   gradient_accumulation_steps=8, logging_strategy="steps", logging_steps=1,
                   logging_first_step=True, disable_tqdm=True, report_to="none")
trainer = DPOTrainer(model=MODEL, args=config, train_dataset=train, peft_config=peft_config)
```

With a LoRA adapter the reference model is the base model with the adapter disabled, so no second copy is loaded.

## GRPO

- Dataset column: "prompt". Reward functions receive the completions (and dataset columns as keyword arguments) and return one float per completion.
- Generation dominates the cost: keep num_generations small (4) and max_completion_length short on small GPUs.
- Shape:

```python
from trl import GRPOConfig, GRPOTrainer

def reward_len(completions, **kwargs):
    return [-abs(200 - len(c if isinstance(c, str) else c[0]["content"])) / 200 for c in completions]

config = GRPOConfig(output_dir="outputs", num_generations=4, max_completion_length=256,
                    per_device_train_batch_size=4, learning_rate=1e-6, logging_strategy="steps",
                    logging_steps=1, logging_first_step=True, disable_tqdm=True, report_to="none")
trainer = GRPOTrainer(model=MODEL, reward_funcs=[reward_len], args=config, train_dataset=train,
                      peft_config=peft_config)
```
