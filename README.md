# PRIME: Plant cis-regulatory element DNA Language Model

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/username/your_model_name)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/Zi-yangChen/PRIME)

PRIME is a **DNA language model** specifically pre-trained on **plant cis-regulatory elements (CREs)**.  
It captures sequence patterns and motif features, enabling applications in:
- Identification of cis-regulatory elements in plant genomes
- Motif feature extraction
- Sequence function prediction
- De novo design regulatory element design

> The model is also available on Hugging Face: [PRIME_Single](https://huggingface.co/Zi-yangChen/PRIME_Single) and [PRIME_BPE](https://huggingface.co/Zi-yangChen/PRIME_BPE)

---

## 📦 Environment Setup

### 1. Clone this repository
```bash
git clone https://github.com/Zi-yangChen/PRIME.git
cd PRIME
```

### 2. Create a Conda environment
```bash
conda create -n prime python=3.9 -y
conda activate prime
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Pre-training from scratch
```bash
python ./run_clm_noblock.py \
    --model_type "gptneo" \
    --tokenizer_name "path/to/tokenizer" \
    --trust_remote_code True \
    --train_file "path/to/train_data" \
    --validation_file "path/to/val_data" \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir "path/to/output" \
    --config_name "path/to/config" \
    --num_train_epochs 10 \
    --save_steps 10000 \
    --block_size 1024
```

### 2. Fine-tuning

