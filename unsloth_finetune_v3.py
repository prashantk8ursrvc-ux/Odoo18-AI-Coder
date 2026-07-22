import os
# Force unsloth fused cross entropy to target 1GB chunk limit.
# This prevents "negligible GPU memory available" error caused by torch's pre-reserved VRAM on Windows.
os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "1"
# Disable torch.compile for cross entropy to prevent massive compilation memory overhead on Windows.
os.environ["UNSLOTH_FUSED_CE_COMPILE_DISABLE"] = "1"

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_FILE = "odoo18_sft_v3.jsonl"
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit" # 4-bit quantized for 8GB VRAM
MAX_SEQ_LENGTH = 4096 # Adjust if needed based on VRAM limitations
OUTPUT_DIR = "odoo18_coder_lora_v3"

# ==========================================
# PROMPT FORMATTING (ChatML format for Qwen)
# ==========================================
def format_prompts(examples):
    texts = []
    # examples["messages"] is a list of lists of dictionaries (since it's batched)
    for messages in examples["messages"]:
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        texts.append(text.strip()) # Remove trailing newline
    return { "text" : texts }

def main():
    print("Loading Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto detection
        load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
    )

    print("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Suggested: 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    print("Loading and Formatting Dataset...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
    
    # Shuffle the dataset to mix the different tiers (MICRO, MESO, MACRO, DOC_GROUNDED)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(format_prompts, batched = True)

    print(f"Dataset Size: {len(dataset)} examples")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 1,  # Reduced to 1 for 8GB VRAM
            gradient_accumulation_steps = 4, # Effective batch size = 4
            warmup_steps = 50,
            num_train_epochs = 1, # Training for 1 epoch
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            save_strategy = "no", # Disable intermediate checkpoints to avoid SFTConfig pickling error
            optim = "paged_adamw_8bit", # Use paged optimizer for VRAM offloading
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use wandb or tensorboard if desired
            gradient_checkpointing = True, # Crucial to enable gradient checkpointing!
            gradient_checkpointing_kwargs = {"use_reentrant": False},
        ),
    )

    print("Starting Training...")
    trainer_stats = trainer.train()
    
    print("Training Completed! Saving LoRA adapters...")
    model.save_pretrained(OUTPUT_DIR) # Local saving
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
