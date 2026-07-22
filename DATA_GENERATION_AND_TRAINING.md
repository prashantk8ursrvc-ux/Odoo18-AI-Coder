# 📊 Data Generation, Fine-Tuning, & GGUF Conversion Guide

This document details exactly how to run the data pipeline, train the Odoo 18 AI Coder using QLoRA, and convert the output to a fast local GGUF model for Ollama.

## 1. Data Generation Pipeline (`data_pipeline/`)

To fine-tune a model, you first need a perfectly formatted dataset. We broke this down into a modular generation pipeline.

### Step 1.1: Generate the Micro & Meso Tier Dataset
Run `generate_micro_dataset.py` to create atomic examples (like `@api.depends` snippets, `<list>` views, `<kanban>` templates).
```bash
python data_pipeline/generate_micro_dataset.py
```
*Outputs*: Intermediate JSON files containing specialized Odoo 18 syntax rules.

### Step 1.2: General Dataset Generation
Run `generate_dataset.py` to pull in Macro (multi-file architectures) and Doc-Grounded data.
```bash
python data_pipeline/generate_dataset.py
```
*Outputs*: Expanded dataset JSON files.

### Step 1.3: Compile & Format for ChatML
To train the AI, the data must be strictly formatted with `<|im_start|>` and `<|im_end|>` ChatML tokens.
```bash
python data_pipeline/compile_odoo_dataset.py
```
*Outputs*: `odoo18_sft_v3.jsonl` (The final, shuffled, formatted dataset ready for training).

### Step 1.4: Validation & Quality Checks
To ensure there are no infinite loops (missing end tokens) or sequence length overflow errors:
```bash
python data_pipeline/analyze_dataset.py
python data_pipeline/check_dataset_quality.py
```

---

## 2. Fine-Tuning with Unsloth

Once `odoo18_sft_v3.jsonl` is ready, it's time to train the LoRA adapter using **Unsloth**, which speeds up training by 2x-5x and cuts VRAM usage by 60%.

### Execution Command:
```bash
python unsloth_finetune_v3.py
```

### What this script does:
1. Loads the base `qwen2.5-coder:7b` model in 4-bit NormalFloat (NF4).
2. Attaches a trainable LoRA adapter (Rank=16, Alpha=32) targeting attention matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.).
3. Runs the Paged AdamW 8-bit optimizer.
4. **Outputs**: When training completes, it saves the adapter to the directory `./odoo18_coder_lora_v3/`. This folder contains `adapter_model.safetensors` (the learned weights) and `adapter_config.json`.

---

## 3. Fast GGUF Conversion

PyTorch models (`.safetensors`) are heavy and require a massive Python environment to run. To run our AI efficiently locally (via Ollama or Llama.cpp), we must convert it to a **GGUF** binary format.

Instead of downloading 16GB of base model weights and slowly merging them in Python, we use a rapid standalone converter that transforms *only* the adapter weights into GGUF.

### Execution Command:
```bash
python convert_lora_to_gguf.py odoo18_coder_lora_v3 --outfile odoo18_coder_lora_v3.gguf
```

### What this script does:
1. Reads the PyTorch LoRA matrices from the `./odoo18_coder_lora_v3/` directory.
2. Converts the FP16/BF16 tensors directly into a memory-mapped `Q4_K_M` GGUF binary format.
3. **Outputs**: A standalone `odoo18_coder_lora_v3.gguf` file (approx. 160 MB) in about **5 seconds**.

---

## 4. Registering the Model with Ollama

Now that we have the lightning-fast `.gguf` adapter, we attach it to the base model inside Ollama using a `Modelfile`.

### The Modelfile Content:
```dockerfile
FROM qwen2.5-coder:7b-instruct
ADAPTER ./odoo18_coder_lora_v3.gguf
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.9
```

### Execution Command:
Run this command from the root of the project to create the local model:
```bash
ollama create odoo18-coder-v3 -f Modelfile
```

Once completed, the model is registered in Ollama. The `Forge` engine will automatically connect to it and start generating production-ready Odoo 18 code!
