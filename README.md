# PRIME: A Generative Foundational Language Model for Plant Cis-Regulatory Elements 

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Zi-yangChen/PRIME_Single)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/Zi-yangChen/PRIME)

PRIME is a **DNA language model** specifically pre-trained on **plant cis-regulatory elements (CREs)**.  
It captures sequence patterns and motif features, enabling applications in:
- Identification of cis-regulatory elements in plant genomes
- Motif feature extraction
- Sequence function prediction
- De novo design regulatory elements

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

If you want to fine-tune PRIME on your own data, prepare a TSV file that contains at least two columns: sequence and label.

```text
sequence    label
ACGTACGT... 1
TGCA...     0
```

Run:

```bash
python finetune.py --config config_for_prediction.json
```


### 3. De novo design

If you want to design CREs with PRIME, prepare a plain-text sequence file (one sequence per line).

Run:

```bash
python generate.py --config config_for_generation.json
```


### 4. Perturbation analysis

Use `explain_relative.py` for perturbation-based interpretability to reveal internal features of CREs.

```bash
python perturb.py \
  --lora_adapter_path path/to/lora_adapter \
  --base_model_path path/to/pretrained_model \
  --data_file_path path/to/data.txt \
  --output_dir path/to/output \
  --kmer_size 3 \
  --context_window_half_size 7 \
  --step_size 1 \
  --max_seq_length_model 1024 \
  --batch_size_pred 32
```
In this script:  
1. `--lora_adapter_path` — Path to the LoRA adapter weights produced by fine-tuning.
2. `--base_model_path` — Path to the pretrained PRIME model weights to load as the base.
3. `--data_file_path` —Input sequences for perturbation, provided as a plain-text file.
4. `--output_dir` — Directory where the output will be saved.
5. `--kmer_size` — Size of the k-mer used when substituting during perturbation.
6. `--context_window_half_size` — Half of the local window size around the perturbed position.
7. `--step_size` — Stride for sliding the perturbation window along the sequence.
8. `--max_seq_length_model` — Maximum sequence length supported by the model during inference.
9. `--batch_size_pred` — Batch size used during prediction.


