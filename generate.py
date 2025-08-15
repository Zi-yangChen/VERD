# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd
import random
import numpy as np
import re
import json
import argparse
import shutil

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DNADataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128, config=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.config = config
        processed_count = 0
        skipped_empty = 0
        skipped_invalid_token = 0

        for text in texts:
            if text and isinstance(text, str):
                tokenized_text = self.safe_encode(text)
                if tokenized_text:
                    self.examples.append(torch.tensor(tokenized_text[:block_size], dtype=torch.long))
                    processed_count +=1
                else:
                    if config and config.get('verbose_logging', False):
                        print(f"Skipping invalid sequence (contains unknown tokens or encoding error): {text[:50]}...")
                    skipped_invalid_token += 1
            else:
                if config and config.get('verbose_logging', False):
                    print(f"Skipping invalid or non-string text: {text}")
                skipped_empty += 1
        
        print(f"DNADataset: Processed {processed_count} sequences. Skipped {skipped_empty} empty/invalid type, {skipped_invalid_token} due to token issues.")
        if not self.examples and processed_count == 0:
            print("Warning: DNADataset is empty after processing. Check your data and tokenizer. No valid sequences found.")

    def safe_encode(self, text):
        try:
            encoded_output = self.tokenizer.encode(text.upper(), add_special_tokens=True)
            return encoded_output
        except Exception as e:
            if self.config and self.config.get('verbose_logging', False):
                print(f"Error encoding sequence: {text[:50]}..., Error: {e}")
            return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def collate_fn_causal(batch, padding_value):
    batch_padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=padding_value)
    return batch_padded

def load_sequences_from_file(file_path, config):
    try:
        data_params = config.get('data_params', {})
        sequence_col_index = data_params.get('sequence_col_index', 0)

        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return []

        is_tsv = False
        if file_path.endswith('.tsv'):
            is_tsv = True
        elif file_path.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f_check:
                    first_line = f_check.readline()
                    if '\t' in first_line:
                        is_tsv = True
            except Exception as e_read_check:
                print(f"Warning: Could not read first line of {file_path} to check for TSV format: {e_read_check}")
        
        if is_tsv:
            df = pd.read_csv(file_path, sep='\t', header=None, usecols=[sequence_col_index], dtype=str, keep_default_na=False, na_filter=False)
            texts = df.iloc[:, 0].tolist()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(texts)} sequences from {file_path}")
        return texts
    except pd.errors.EmptyDataError:
        print(f"Warning: File {file_path} is empty or has no columns to parse.")
        return []
    except ValueError as ve:
        print(f"Error processing file {file_path} with pandas (likely bad 'sequence_col_index' or file format): {ve}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred loading data from {file_path}: {e}")
        return []

class LengthStoppingCriteria(StoppingCriteria):
    def __init__(self, max_total_length):
        self.max_total_length = max_total_length

    def __call__(self, input_ids, scores):
        return input_ids.shape[1] >= self.max_total_length

def generate_sequences_with_lora(model, tokenizer, config_generate, device):
    model.eval()
    all_generated_sequences = []
    
    prompt_text = config_generate.get("prompt", "")
    input_ids_list = []
    if not prompt_text:
        print("Warning: No prompt provided for generation. Using BOS token if available, else empty input.")
        if tokenizer.bos_token_id is not None:
            input_ids_list = [tokenizer.bos_token_id]
        input_ids = tokenizer.encode(tokenizer.decode(input_ids_list) if input_ids_list else "", return_tensors='pt').to(device)
    else:
        input_ids = tokenizer.encode(prompt_text.upper(), return_tensors='pt').to(device)

    num_sequences_to_generate = config_generate.get("num_sequences", 10)
    
    # max_length is total length (prompt + new tokens)
    # max_new_tokens specifies only the number of new tokens to generate
    # If both are set, max_new_tokens usually takes precedence in model.generate().
    max_new_tokens_val = config_generate.get("max_new_tokens")
    gen_max_length_total = config_generate.get("max_length")

    if max_new_tokens_val is not None:
        if gen_max_length_total is not None:
            print(f"Info: Both 'max_new_tokens' ({max_new_tokens_val}) and 'max_length' ({gen_max_length_total}) are set. "
                  f"'max_new_tokens' will be primarily used by model.generate().")
    elif gen_max_length_total is not None:
        max_new_tokens_val = gen_max_length_total - input_ids.shape[1]
        if max_new_tokens_val <= 0:
            print(f"Warning: max_length ({gen_max_length_total}) is not greater than prompt length ({input_ids.shape[1]}). "
                  f"Setting max_new_tokens to a default of 50.")
            max_new_tokens_val = 50
    else:
        print("Warning: Neither 'max_new_tokens' nor 'max_length' found in generation_params. Using default max_new_tokens=50.")
        max_new_tokens_val = 50
        gen_max_length_total = input_ids.shape[1] + max_new_tokens_val

    # min_length refers to the total sequence length (prompt + generated)
    gen_min_length_total = config_generate.get("min_length", 10)
    if gen_min_length_total > (input_ids.shape[1] + max_new_tokens_val):
        print(f"Warning: min_length ({gen_min_length_total}) is greater than effective max_length "
              f"({input_ids.shape[1] + max_new_tokens_val}). Adjusting min_length.")
        gen_min_length_total = input_ids.shape[1] + max_new_tokens_val

    print(f"Generating {num_sequences_to_generate} sequences with prompt: '{prompt_text}' (Prompt length: {input_ids.shape[1]})")
    print(f"Effective generation params: max_new_tokens={max_new_tokens_val}, min_total_length={gen_min_length_total}, "
          f"temp={config_generate.get('temperature', 0.8)}, top_p={config_generate.get('top_p', 0.9)}")

    for i in range(num_sequences_to_generate):
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids if input_ids.nelement() > 0 else None,
                max_new_tokens=max_new_tokens_val,
                min_length=gen_min_length_total,
                temperature=config_generate.get("temperature", 0.8),
                top_p=config_generate.get("top_p", 0.9),
                top_k=config_generate.get("top_k", 50),
                repetition_penalty=config_generate.get("repetition_penalty", 1.2),
                do_sample=config_generate.get("do_sample", True),
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            for generated_sequence_ids in output_sequences:
                # Determine start index for decoding (after prompt)
                decode_start_index = 0
                if input_ids.nelement() > 0 and input_ids.shape[1] > 0:
                    if torch.equal(generated_sequence_ids[:input_ids.shape[1]], input_ids[0]):
                        decode_start_index = input_ids.shape[1]
                
                generated_ids_only = generated_sequence_ids[decode_start_index:]
                decoded_sequence = tokenizer.decode(generated_ids_only, skip_special_tokens=True)
                cleaned_sequence = re.sub(r'[^ATCG]', '', decoded_sequence.upper())
                
                if cleaned_sequence:
                    all_generated_sequences.append(cleaned_sequence)
                elif decoded_sequence and config_generate.get('verbose_logging', False):
                    print(f"  Sequence became empty after ATCG cleaning. Original decoded (new part): '{decoded_sequence[:50]}...'")
        
        if num_sequences_to_generate > 0 and (i + 1) % max(1, num_sequences_to_generate // 10) == 0:
            print(f"  Generated {i+1}/{num_sequences_to_generate} sequences...")

    return all_generated_sequences

def save_sequences_to_output_file(sequences, file_path):
    try:
        with open(file_path, 'w') as f:
            for seq in sequences:
                f.write(seq + '\n')
        print(f"Successfully saved {len(sequences)} generated sequences to {file_path}")
    except Exception as e:
        print(f"Error saving sequences to {file_path}: {e}")

def main(config_data, original_config_path):
    set_seed(config_data['training_params'].get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = config_data['output_params']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    try:
        saved_config_filename = "training_run_config.json"
        destination_config_path = os.path.join(output_dir, saved_config_filename)
        shutil.copy2(original_config_path, destination_config_path)
        print(f"Copied original config file '{original_config_path}' to '{destination_config_path}'")
    except Exception as e_copy:
        print(f"Warning: Could not save config file to output directory. Error: {e_copy}")

    print(f"Loading tokenizer from: {config_data['model_params']['tokenizer_name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config_data['model_params']['tokenizer_name_or_path'],
        use_fast=config_data['model_params'].get('use_fast_tokenizer', True)
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set tokenizer.pad_token to eos_token: {tokenizer.eos_token}")
        else:
            print("Warning: tokenizer.pad_token and tokenizer.eos_token are None. Adding a new pad_token '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print(f"Loading base CausalLM model from: {config_data['model_params']['base_model_name_or_path']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config_data['model_params']['base_model_name_or_path'],
        trust_remote_code=config_data['model_params'].get('trust_remote_code', False)
    )
    
    if len(tokenizer) != base_model.config.vocab_size:
        print(f"Warning: Tokenizer vocab size ({len(tokenizer)}) and model vocab size ({base_model.config.vocab_size}) differ.")
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model token embeddings to match tokenizer size: {len(tokenizer)}")
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config_params = config_data['lora_params']
    peft_lora_config = LoraConfig(
        r=lora_config_params['r'],
        lora_alpha=lora_config_params['lora_alpha'],
        target_modules=lora_config_params['target_modules'],
        lora_dropout=lora_config_params['lora_dropout'],
        bias=lora_config_params.get('bias', "none"),
        task_type=TaskType.CAUSAL_LM
    )
    
    lora_model = get_peft_model(base_model, peft_lora_config)
    print("LoRA model created for Causal LM.")
    lora_model.print_trainable_parameters()
    lora_model.to(device)

    print("Loading training and validation data...")
    train_texts = load_sequences_from_file(config_data['data_params']['train_file'], config_data)
    val_texts = load_sequences_from_file(config_data['data_params']['val_file'], config_data)

    block_size = config_data['model_params'].get('block_size', 128)
    
    dataset_init_cfg_dict = {'verbose_logging': config_data.get('verbose_logging', False)}
    train_dataset = DNADataset(train_texts, tokenizer, block_size=block_size, config=dataset_init_cfg_dict)
    val_dataset = DNADataset(val_texts, tokenizer, block_size=block_size, config=dataset_init_cfg_dict)
    
    if not train_dataset or len(train_dataset) == 0:
        print("Error: Training dataset is empty or contains no valid examples. Exiting.")
        return
    if not val_dataset or len(val_dataset) == 0:
        print("Warning: Validation dataset is empty or contains no valid examples. Proceeding without validation.")
        val_loader = None
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config_data['training_params']['batch_size'],
            shuffle=False,
            collate_fn=lambda batch: collate_fn_causal(batch, tokenizer.pad_token_id)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_data['training_params']['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn_causal(batch, tokenizer.pad_token_id)
    )
    
    if len(train_loader) == 0:
        print("Error: Training DataLoader is empty. Cannot proceed with training. Check data loading and DNADataset.")
        return

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, lora_model.parameters()),
        lr=config_data['training_params']['learning_rate'],
        weight_decay=config_data['training_params'].get('weight_decay', 0.01)
    )
    
    num_epochs = config_data['training_params']['num_epochs']
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps_config = config_data['training_params'].get('warmup_steps', 0)
    if isinstance(num_warmup_steps_config, float):
        num_warmup_steps = int(num_warmup_steps_config * num_training_steps)
    else:
        num_warmup_steps = int(num_warmup_steps_config)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print("Starting LoRA fine-tuning for Causal LM...")
    best_val_loss = float('inf')
    best_adapter_path = os.path.join(output_dir, "best_lora_adapter_causal_lm")

    for epoch in range(num_epochs):
        lora_model.train()
        total_train_loss = 0
        batch_iterator = iter(train_loader)
        
        for batch_idx in range(len(train_loader)):
            try:
                batch_input_ids = next(batch_iterator)
            except StopIteration:
                print(f"Warning: train_loader exhausted prematurely at batch {batch_idx} in epoch {epoch+1}. This might indicate issues with dataset length or DataLoader.")
                break

            inputs = batch_input_ids.to(device)
            outputs = lora_model(input_ids=inputs, attention_mask=(inputs != tokenizer.pad_token_id), labels=inputs)
            loss = outputs.loss
            
            if loss is None:
                print(f"Warning: Loss is None at Epoch {epoch+1}, Batch {batch_idx+1}. Skipping batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_train_loss += loss.item()
            if (batch_idx + 1) % config_data['training_params'].get('logging_steps', 50) == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
        
        if len(train_loader) > 0 :
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} - No batches processed in training.")

        if val_loader and len(val_loader) > 0:
            lora_model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_input_ids in val_loader:
                    inputs = batch_input_ids.to(device)
                    outputs = lora_model(input_ids=inputs, attention_mask=(inputs != tokenizer.pad_token_id), labels=inputs)
                    if outputs.loss is not None:
                        total_val_loss += outputs.loss.item()
                    else:
                        print(f"Warning: Validation loss is None for a batch.")

            if len(val_loader) > 0:
                avg_val_loss = total_val_loss / len(val_loader)
                print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    lora_model.save_pretrained(best_adapter_path)
                    tokenizer.save_pretrained(best_adapter_path)
                    print(f"  New best model found! Validation Loss: {best_val_loss:.4f}. Adapter saved to {best_adapter_path}")
            else:
                print(f"Epoch {epoch+1} - Validation loader was empty, skipping validation loss calculation.")
        else:
            print(f"Epoch {epoch+1} - No validation loader or it's empty. Saving model at end of epoch if it's the last one.")
            if epoch == num_epochs - 1 :
                 lora_model.save_pretrained(best_adapter_path)
                 tokenizer.save_pretrained(best_adapter_path)
                 print(f"  Model adapter saved at end of training (epoch {epoch+1}) to {best_adapter_path} (no/empty validation).")

    print("Training finished.")
    if os.path.exists(best_adapter_path):
        print(f"Best/Final LoRA adapter saved to: {best_adapter_path}")
    else:
        print(f"Warning: No LoRA adapter was saved. Check training logs and validation behavior. Path expected: {best_adapter_path}")

    if os.path.exists(best_adapter_path):
        print("\nLoading best/final LoRA adapter for generation...")
        try:
            final_base_model_config = config_data['model_params']['base_model_name_or_path']
            final_base_model = AutoModelForCausalLM.from_pretrained(
                final_base_model_config,
                trust_remote_code=config_data['model_params'].get('trust_remote_code', False)
            )
            if len(tokenizer) != final_base_model.config.vocab_size:
                final_base_model.resize_token_embeddings(len(tokenizer))
            if tokenizer.pad_token_id is not None:
                final_base_model.config.pad_token_id = tokenizer.pad_token_id

            generation_model = PeftModel.from_pretrained(final_base_model, best_adapter_path)
            generation_model = generation_model.to(device)
            
            if config_data.get('generation_params', {}).get('merge_lora_weights_for_generation', False):
                print("Merging LoRA weights for generation model...")
                generation_model = generation_model.merge_and_unload()
                print("LoRA weights merged.")
            
            generation_model.eval()

            if config_data.get("generation_params"):
                print("\nStarting sequence generation...")
                generated_dna_sequences = generate_sequences_with_lora(
                    model=generation_model,
                    tokenizer=tokenizer,
                    config_generate=config_data['generation_params'],
                    device=device
                )
                
                output_gen_file = config_data['output_params'].get("generated_sequences_file", "generated_dna_lora.txt")
                full_output_gen_path = os.path.join(output_dir, output_gen_file)
                save_sequences_to_output_file(generated_dna_sequences, full_output_gen_path)
            else:
                print("No 'generation_params' found in config. Skipping sequence generation.")
        except Exception as e_gen_load:
            print(f"Error during setup for generation (loading model/adapter): {e_gen_load}")
            print("Skipping generation part.")
    else:
        print(f"No adapter found at {best_adapter_path}. Skipping generation.")

    print("\nLoRA Generative Fine-tuning script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Causal Language Models (e.g., DNA sequence generation).")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config_dict_from_file = json.load(f)
        print(f"Configuration loaded successfully from {args.config}")
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        exit(1)
        
    main(config_dict_from_file, args.config)