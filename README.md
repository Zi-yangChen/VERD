# PRIME: Plant cis-regulatory element DNA Language Model

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Zi-yangChen/PRIME_Single)
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
You can use the `run_clm_noblock.py` script to pretrain a model using your own dataset.  
Before training the model, you need a `config.json` file, which you can download from either [PRIME_Single](https://huggingface.co/Zi-yangChen/PRIME_Single) or [PRIME_BPE](https://huggingface.co/Zi-yangChen/PRIME_BPE).

```bash
python ./run_clm_noblock.py \
    --model_type "gptneo" \
    --tokenizer_name "path/to/tokenizer" \
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

In this script:  
1. `--model_type` — Specifies the model architecture; here we use GPT-Neo, an autoregressive language model.
2. `--tokenizer_name` — Path to the tokenizer (local directory or a Hugging Face repo ID).
3. `--train_file` — Path to the training dataset file.
4. `--validation_file` — Path to the validation dataset file.
5. `--learning_rate` — Initial learning rate.
6. `--per_device_train_batch_size` — Batch size per device for training.
7. `--per_device_eval_batch_size` — Batch size per device for evaluation.
8. `--do_train` — Enable the training process.
9. `--do_eval` — Enable evaluation during/after training.
10. `--output_dir` — Directory to save checkpoints, logs, and the final model.
11. `--config_name` — Path (or repo ID) to `config.json` containing the model configuration.
12. `--num_train_epochs` — Number of training epochs. *Alternatively, set `--max_steps` and adjust save/log/eval strategies.*
13. `--save_steps` — Interval (in steps) between checkpoint saves.
14. `--block_size` — Maximum sequence length (in tokens); longer sequences will be truncated.

### 2. Fine-tuning
如果你想基于PRIME预训练模型并利用自己的数据微调，请准备好tsv格式的数据集文件，确保文件中含有sequence列和label列。
```bash
python finetune.py --config/config_for_prediction.json
```

### 3. De novo design
如果你想基于PRIME从头设计CREs，请准备好txt格式的序列文件
```bash
python generate.py --config/config_for_generation.json
```

### 4. Perturbation analysis
你可以使用`perturb.py`脚本进行基于扰动的可解释性分析，发现CREs的内部特征！
