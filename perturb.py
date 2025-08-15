# -*- coding: utf-8 -*-
import json
import argparse
import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel, PeftConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DNA_ALPHABET = ['A', 'T', 'C', 'G']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_random_kmer(k, kmer_to_avoid=None):
    if kmer_to_avoid is None:
        return "".join(random.choice(DNA_ALPHABET) for _ in range(k))
    
    while True:
        random_kmer = "".join(random.choice(DNA_ALPHABET) for _ in range(k))
        if random_kmer != kmer_to_avoid:
            return random_kmer

def perturb_sequence_kmer(original_sequence, start_pos, kmer_size, replacement_kmer):
    s_list = list(original_sequence)
    
    if start_pos + kmer_size > len(s_list):
        return original_sequence

    for i in range(kmer_size):
        s_list[start_pos + i] = replacement_kmer[i]
    return "".join(s_list)

def get_predictions_for_sequences(sequences_list, model, tokenizer, max_seq_length, batch_size_pred):
    model.eval()
    all_predictions = []
    
    class TempDataset(Dataset):
        def __init__(self, texts, tok, max_len):
            self.encodings = tok(texts, truncation=True, padding=False, max_length=max_len)
        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}
        def __len__(self, ):
            return len(self.encodings['input_ids'])

    dataset = TempDataset(sequences_list, tokenizer, max_seq_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
    dataloader = DataLoader(dataset, batch_size=batch_size_pred, collate_fn=collator, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            inputs_on_device = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**inputs_on_device)
            predictions = outputs.logits.squeeze(-1).cpu().numpy()
            all_predictions.extend(predictions)
            
    return np.array(all_predictions)

def run_kmer_perturbation_analysis(
    model, tokenizer, sequences_to_analyze, 
    kmer_size, step_size, 
    max_seq_length_model, batch_size_pred,
    output_dir,
    context_window_half_size,
    expected_seq_length=170
    ):
    
    os.makedirs(output_dir, exist_ok=True)
    num_sequences = len(sequences_to_analyze)
    if num_sequences == 0:
        # Return empty DataFrame if no sequences
        return np.array([]), pd.DataFrame()

    baseline_predictions = get_predictions_for_sequences(sequences_to_analyze, model, tokenizer, max_seq_length_model, batch_size_pred)

    # These are still calculated but will not be plotted
    positional_abs_diff_sum = np.zeros(expected_seq_length)
    positional_perturb_counts = np.zeros(expected_seq_length, dtype=int)
    
    perturbation_log = []

    for seq_idx, original_sequence in enumerate(sequences_to_analyze):
        if len(original_sequence) != expected_seq_length:
            continue

        baseline_pred_for_seq = baseline_predictions[seq_idx]
        
        perturbed_sequences_batch = []
        perturbation_info_batch = []

        for start_pos in range(0, len(original_sequence) - kmer_size + 1, step_size):
            original_kmer = original_sequence[start_pos : start_pos + kmer_size]
            
            replacement_kmer = generate_random_kmer(kmer_size, kmer_to_avoid=original_kmer)
            
            perturbed_seq = perturb_sequence_kmer(original_sequence, start_pos, kmer_size, replacement_kmer)
            perturbed_sequences_batch.append(perturbed_seq)
            perturbation_info_batch.append({
                'original_kmer': original_kmer, 
                'replacement_kmer': replacement_kmer,
                'start_pos': start_pos
            })

        if not perturbed_sequences_batch:
            continue

        perturbed_predictions = get_predictions_for_sequences(perturbed_sequences_batch, model, tokenizer, max_seq_length_model, batch_size_pred)

        for i, info in enumerate(perturbation_info_batch):
            start_pos = info['start_pos']
            
            prediction_diff = perturbed_predictions[i] - baseline_pred_for_seq
            
            # Positional importance calculation still happens
            abs_diff = np.abs(prediction_diff)
            for pos_in_kmer in range(kmer_size):
                actual_seq_pos = start_pos + pos_in_kmer
                if 0 <= actual_seq_pos < expected_seq_length:
                    positional_abs_diff_sum[actual_seq_pos] += abs_diff
                    positional_perturb_counts[actual_seq_pos] += 1
            
            context_start = max(0, start_pos - context_window_half_size)
            context_end = min(len(original_sequence), start_pos + kmer_size + context_window_half_size)
            context_sequence = original_sequence[context_start:context_end]
            
            position_relative_to_tss = start_pos - 165

            perturbation_log.append({
                'sequence_index': seq_idx,
                'position_relative_to_TSS': position_relative_to_tss,
                'original_kmer': info['original_kmer'],
                'perturbed_kmer': info['replacement_kmer'],
                'context_sequence': context_sequence,
                'prediction_difference': prediction_diff
            })

    # Finalize Positional Importance (still calculated, just not plotted)
    avg_positional_importance = np.zeros_like(positional_abs_diff_sum)
    non_zero_counts_mask = positional_perturb_counts > 0
    avg_positional_importance[non_zero_counts_mask] = positional_abs_diff_sum[non_zero_counts_mask] / positional_perturb_counts[non_zero_counts_mask]


    # Save Perturbation Log to CSV
    if perturbation_log:
        df_log = pd.DataFrame(perturbation_log)
        df_log = df_log[[
            'sequence_index', 
            'position_relative_to_TSS', 
            'context_sequence',
            'original_kmer',
            'perturbed_kmer',
            'prediction_difference'
        ]]
        csv_path = os.path.join(output_dir, f"perturbation_log_k{kmer_size}.csv")
        df_log.to_csv(csv_path, index=False)
    else:
        pass # No data to log

    # Return the main results (positional importance array and detailed log DataFrame)
    return avg_positional_importance, pd.DataFrame(perturbation_log)

def load_sequences_from_file(file_path, config):
    try:
        data_params = config.get('data_params', {})
        sequence_col_index = data_params.get('sequence_col_index', 0)

        if not os.path.exists(file_path):
            return []

        is_tsv = file_path.lower().endswith('.tsv')
        
        if is_tsv:
            df = pd.read_csv(file_path, sep='\t', header=None, usecols=[sequence_col_index], dtype=str, keep_default_na=False, na_filter=False)
            texts = df.iloc[:, 0].tolist() 
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip().upper() for line in f if line.strip()]
        
        return texts
    except Exception as e:
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-mer perturbation analysis for DNA sequence regression models.")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path to the saved LoRA adapter directory.")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to the base Hugging Face model. If not provided, attempts to infer from LoRA config.")
    parser.add_argument("--data_file_path", type=str, required=True, help="Path to the TSV or TXT file containing sequences to analyze.")
    parser.add_argument("--sequence_col_index_in_tsv", type=int, default=0, help="If data_file_path is TSV, 0-based index of the column with sequences.")
    parser.add_argument("--output_dir", type=str, default="./perturbation_analysis_regression", help="Directory to save analysis outputs.")
    
    parser.add_argument("--kmer_size", type=int, default=1, help="Size of k-mers for perturbation. Set to 1 for single-base-pair analysis.")
    parser.add_argument("--step_size", type=int, default=1, help="Step size for sliding k-mer window.")
    parser.add_argument("--context_window_half_size", type=int, default=10, help="Size of the window on each side of the k-mer to record in the log (x). Total context is k+2x.")
    
    parser.add_argument("--expected_seq_length", type=int, default=170, help="Expected length of sequences for positional analysis.")
    
    parser.add_argument("--max_seq_length_model", type=int, default=512, help="Maximum sequence length model can handle (for tokenizer).")
    parser.add_argument("--batch_size_pred", type=int, default=32, help="Batch size for making predictions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Set if your base model requires trust_remote_code=True.")

    args = parser.parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)

    base_model_name_or_path_resolved = None
    try:
        peft_config = PeftConfig.from_pretrained(args.lora_adapter_path)
        if hasattr(peft_config, 'base_model_name_or_path') and peft_config.base_model_name_or_path:
            base_model_name_or_path_resolved = peft_config.base_model_name_or_path
    except Exception: pass

    if not base_model_name_or_path_resolved:
        if args.base_model_path:
            base_model_name_or_path_resolved = args.base_model_path
        else:
            exit(1)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path_resolved)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name_or_path_resolved,
        num_labels=1,
        problem_type="regression",
        trust_remote_code=args.trust_remote_code
    )
    if len(tokenizer) > model.config.vocab_size: model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model, args.lora_adapter_path)
    model = model.to(DEVICE)
    model.eval()

    dummy_data_config = {'data_params': {'sequence_col_index': args.sequence_col_index_in_tsv}}
    sequences_to_analyze = load_sequences_from_file(args.data_file_path, dummy_data_config)
    
    if not sequences_to_analyze:
        exit(1)

    run_kmer_perturbation_analysis(
        model, tokenizer, sequences_to_analyze,
        kmer_size=args.kmer_size,
        step_size=args.step_size,
        max_seq_length_model=args.max_seq_length_model,
        batch_size_pred=args.batch_size_pred,
        output_dir=args.output_dir,
        expected_seq_length=args.expected_seq_length,
        context_window_half_size=args.context_window_half_size
    )