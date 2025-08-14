import json
import argparse
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset, Features, Value, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig

# --- Global Variables ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helper Functions ---
def compute_metrics_classification(eval_pred, config):
    logits, labels = eval_pred
    task_params = config['task_params']

    if task_params['num_classes'] == 1:  # Binary classification with single logit (e.g., sigmoid output)
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy().squeeze()
        predictions_binary = (probs >= 0.5).astype(int)
        probs_positive_class = probs
    else:  # Binary (two logits) or Multiclass classification
        probs_all_classes = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        predictions_binary = np.argmax(probs_all_classes, axis=1)
        if task_params['num_classes'] == 2:
            probs_positive_class = probs_all_classes[:, 1]
        else:  # For multiclass, roc_auc_score handles 'ovr' with full probabilities
            probs_positive_class = probs_all_classes

    labels = np.array(labels)
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, predictions_binary)

    if task_params['num_classes'] == 2 or task_params['num_classes'] == 1:  # Binary specific metrics
        metrics['precision'] = precision_score(labels, predictions_binary, zero_division=0)
        metrics['recall'] = recall_score(labels, predictions_binary, zero_division=0)
        metrics['f1'] = f1_score(labels, predictions_binary, zero_division=0)
        try:
            metrics['auc'] = roc_auc_score(labels, probs_positive_class)
        except ValueError as e:
            metrics['auc'] = np.nan
            print(f"Warning: AUC calculation failed for classification: {e}")
    elif task_params['num_classes'] > 2:  # Multiclass metrics
        metrics['f1_macro'] = f1_score(labels, predictions_binary, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(labels, predictions_binary, average='micro', zero_division=0)
        try:
            metrics['roc_auc_ovr_macro'] = roc_auc_score(labels, probs_positive_class, average='macro',
                                                         multi_class='ovr')
        except ValueError as e:
            metrics['roc_auc_ovr_macro'] = np.nan
            print(f"Warning: Multiclass OVR AUC calculation failed: {e}")
    return metrics


def compute_metrics_regression(eval_pred, config=None):
    predictions, labels = eval_pred
    labels = np.array(labels).squeeze()
    predictions = np.array(predictions).squeeze()

    metrics = {}
    metrics['r2'] = r2_score(labels, predictions)
    metrics['mse'] = mean_squared_error(labels, predictions)
    metrics['mae'] = mean_absolute_error(labels, predictions)

    if len(np.unique(labels)) > 1 and len(np.unique(predictions)) > 1:
        spearman_corr, _ = spearmanr(labels, predictions)
        pearson_corr, _ = pearsonr(labels, predictions)
    else:  # Cannot compute correlation if labels or predictions are constant
        spearman_corr, pearson_corr = np.nan, np.nan
        print("Warning: Cannot compute Spearman/Pearson correlation due to constant labels or predictions.")

    metrics['spearmanr'] = spearman_corr
    metrics['pearsonr'] = pearson_corr
    return metrics


def plot_classification_results(true_labels, probabilities, output_dir, config, prefix="test"):
    os.makedirs(output_dir, exist_ok=True)
    task_params = config['task_params']

    if task_params['num_classes'] == 1 or task_params[
        'num_classes'] == 2:  # Binary classification (single logit or two logits)
        probs_positive_class = probabilities  # If single logit, probabilities is already prob of positive class. If 2 logits, it should be probabilities_all[:, 1]
        predictions_binary = (probs_positive_class >= 0.5).astype(int)

        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, probs_positive_class)
        auc_score_val = roc_auc_score(true_labels, probs_positive_class)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score_val:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(
            f'{prefix.capitalize()} Set ROC Curve\nModel: {os.path.basename(config["model_params"]["base_model_name_or_path"])}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{prefix}_roc_curve.png"), dpi=300)
        plt.close()
        print(f"{prefix.capitalize()} ROC curve saved to {os.path.join(output_dir, f'{prefix}_roc_curve.png')}")

        # Confusion Matrix for binary
        cm = confusion_matrix(true_labels, predictions_binary)
        plt.figure(figsize=(7, 5))
        class_names_display = ['Class 0', 'Class 1']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names_display,
                    yticklabels=class_names_display)
        plt.title(f'{prefix.capitalize()} Set Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"), dpi=300)
        plt.close()
        print(
            f"{prefix.capitalize()} Confusion Matrix saved to {os.path.join(output_dir, f'{prefix}_confusion_matrix.png')}")

    elif task_params['num_classes'] > 2:  # Multiclass
        predictions_multiclass = np.argmax(probabilities,
                                           axis=1)  # probabilities here is probs_all_classes (N x num_classes)

        # Confusion Matrix for multiclass
        cm = confusion_matrix(true_labels, predictions_multiclass)
        plt.figure(figsize=(max(7, task_params['num_classes']), max(5, task_params['num_classes'] * 0.8)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f"Class {i}" for i in range(task_params['num_classes'])],
                    yticklabels=[f"Class {i}" for i in range(task_params['num_classes'])])
        plt.title(f'{prefix.capitalize()} Set Confusion Matrix (Multiclass)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix_multiclass.png"), dpi=300)
        plt.close()
        print(
            f"{prefix.capitalize()} Multiclass Confusion Matrix saved to {os.path.join(output_dir, f'{prefix}_confusion_matrix_multiclass.png')}")


def plot_regression_results(true_labels, predictions, output_dir, config, prefix="test"):
    os.makedirs(output_dir, exist_ok=True)
    true_labels = np.array(true_labels).squeeze()
    predictions = np.array(predictions).squeeze()

    # Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(true_labels, predictions, alpha=0.5, label="Predictions vs True", s=10)

    # Determine plot limits dynamically
    if len(true_labels) > 0 and len(predictions) > 0:
        all_values = np.concatenate([true_labels, predictions])
        min_val = np.min(all_values) - np.std(all_values) * 0.1
        max_val = np.max(all_values) + np.std(all_values) * 0.1
        if min_val == max_val:  # Handle case where all values are the same
            min_val -= 0.5
            max_val += 0.5
    else:
        min_val, max_val = 0, 1

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal y=x line")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(
        f'{prefix.capitalize()} Set: True vs. Predicted Values\nModel: {os.path.basename(config["model_params"]["base_model_name_or_path"])}')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    metrics_text_scatter = []
    if len(true_labels) > 1 and len(predictions) > 1:
        r2 = r2_score(true_labels, predictions)
        metrics_text_scatter.append(f'R² = {r2:.3f}')
        if len(np.unique(true_labels)) > 1 and len(np.unique(predictions)) > 1:
            spearman_val, _ = spearmanr(true_labels, predictions)
            pearson_val, _ = pearsonr(true_labels, predictions)
            metrics_text_scatter.append(f'Spearman ρ = {spearman_val:.3f}')
            metrics_text_scatter.append(f'Pearson r = {pearson_val:.3f}')
        else:
            metrics_text_scatter.append('Spearman ρ = N/A')
            metrics_text_scatter.append('Pearson r = N/A')

        plt.text(0.05, 0.95, "\n".join(metrics_text_scatter),
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_regression_scatter.png"), dpi=300)
    plt.close()
    print(
        f"{prefix.capitalize()} Regression scatter plot saved to {os.path.join(output_dir, f'{prefix}_regression_scatter.png')}")

    # 2D Kernel Density Estimate (KDE) Plot with Dark Background
    if len(true_labels) > 1 and len(predictions) > 1 and \
            len(np.unique(true_labels)) > 1 and len(np.unique(predictions)) > 1:

        fig, kde_ax = plt.subplots(figsize=(8, 7))

        # Determine plot limits dynamically
        if len(true_labels) > 0 and len(predictions) > 0:
            all_values_kde = np.concatenate([true_labels, predictions])
            margin_factor = 0.2
            min_val_kde = np.min(all_values_kde) - np.std(all_values_kde) * margin_factor
            max_val_kde = np.max(all_values_kde) + np.std(all_values_kde) * margin_factor
            if min_val_kde == max_val_kde:
                min_val_kde -= 0.5
                max_val_kde += 0.5
        else:
            min_val_kde, max_val_kde = 0, 1

        # Set a uniform dark purple background for the axes area
        kde_ax.set_facecolor('#582A72')  # Deep purple

        sns.kdeplot(
            x=true_labels,
            y=predictions,
            ax=kde_ax,
            fill=True,
            cmap="Purples_r",  # High density = Lighter Purple/White
            levels=15,
            thresh=0.005,  # Controls which density levels are filled
            alpha=0.85
        )

        sns.kdeplot(
            x=true_labels,
            y=predictions,
            ax=kde_ax,
            levels=7,
            color='white',
            linewidths=0.7,
            linestyles='--',
            alpha=0.6
        )

        kde_ax.plot([min_val_kde, max_val_kde], [min_val_kde, max_val_kde], 'k--', lw=1.5, label="Ideal y=x line")

        kde_ax.set_xlabel("True Values")
        kde_ax.set_ylabel("Predicted Values")
        title_text = f'{prefix.capitalize()} Set: True vs. Predicted KDE'
        model_name_short = os.path.basename(config["model_params"]["base_model_name_or_path"])
        kde_ax.set_title(f'{title_text}\nModel: {model_name_short}')

        kde_ax.set_xlim(min_val_kde, max_val_kde)
        kde_ax.set_ylim(min_val_kde, max_val_kde)

        spearman_val_kde, _ = spearmanr(true_labels, predictions)

        kde_ax.text(0.04, 0.96, f'Spearman ρ = {spearman_val_kde:.4f}',
                    transform=kde_ax.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='black', lw=1, alpha=0.8))

        kde_ax.grid(True, linestyle=':', alpha=0.3, color='white')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_regression_kde_v2.png"), dpi=300)
        plt.close(fig)
        print(
            f"{prefix.capitalize()} Regression KDE plot (v2) saved to {os.path.join(output_dir, f'{prefix}_regression_kde_v2.png')}")
    else:
        print(f"Skipping KDE plot for {prefix} set due to insufficient data variance or too few points.")


# --- Main Function ---
def main(config_path):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Configuration loaded:")
    print(json.dumps(config, indent=4))

    output_dir = config['output_params']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    try:
        config_filename = os.path.basename(config_path)
        shutil.copy2(config_path, os.path.join(output_dir, config_filename))
        print(f"Copied config file to {os.path.join(output_dir, config_filename)}")
    except Exception as e:
        print(f"Warning: Could not copy config file to output directory. Error: {e}")

    task_type = config['task_params']['type']
    num_classes = config['task_params'].get('num_classes')
    if task_type == "classification" and num_classes is None:
        raise ValueError("'num_classes' must be specified in 'task_params' for classification tasks.")

    num_labels_for_model = 1 if task_type == "regression" else num_classes

    # 2. Load Tokenizer
    print(f"Loading tokenizer from: {config['model_params']['tokenizer_name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_params']['tokenizer_name_or_path'])
    if tokenizer.pad_token is None:  # Ensure tokenizer has a pad token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set tokenizer.pad_token to eos_token: {tokenizer.pad_token}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print(f"Added new pad_token: {tokenizer.pad_token}")

    # 3. Load Data using Hugging Face datasets
    print("Loading and preprocessing data using Hugging Face datasets...")
    data_files = {
        "train": config['data_params']['train_file'],
        "validation": config['data_params']['val_file'],
        "test": config['data_params']['test_file']
    }

    sequence_col_name = config['data_params']['sequence_col']
    label_col_name = config['data_params']['label_col']

    dataset_features = {sequence_col_name: Value('string')}
    if task_type == "classification":
        dataset_features[label_col_name] = Value('string')  # Load as string first, then convert to int
    elif task_type == "regression":
        dataset_features[label_col_name] = Value('float32')

    try:
        raw_datasets = load_dataset('csv', data_files=data_files, delimiter='\t', features=Features(dataset_features))
    except Exception as e:
        print(f"Error loading dataset with explicit features: {e}. Trying inference.")
        raw_datasets = load_dataset('csv', data_files=data_files, delimiter='\t')

    max_length = config['model_params'].get('max_seq_length', 512)

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(
            [str(s).upper() for s in examples[sequence_col_name]],
            truncation=True,
            padding=False,  # DataCollatorWithPadding handles batch-wise padding
            max_length=max_length
        )

        if task_type == "classification":
            try:
                labels_as_int = [int(l) for l in examples[label_col_name]]
                # Basic check for label range (0 to num_classes-1). More robust mapping might be needed.
                if not all(0 <= lbl < num_classes for lbl in labels_as_int):
                    unique_batch_labels = sorted(list(set(labels_as_int)))
                    if len(unique_batch_labels) == num_classes:
                        label_map_local = {orig: i for i, orig in enumerate(unique_batch_labels)}
                        labels_as_int = [label_map_local[l_int] for l_int in labels_as_int]
                    else:
                        print(f"Warning: Batch labels {unique_batch_labels} don't map directly to 0-{num_classes - 1}.")
                tokenized_inputs['labels'] = labels_as_int
            except Exception as e_label:
                print(f"Error processing labels for classification: {e_label}. Labels: {examples[label_col_name][:5]}")
                raise  # Re-raise to indicate critical data processing failure
        elif task_type == "regression":
            tokenized_inputs['labels'] = [float(l) for l in examples[label_col_name]]
        return tokenized_inputs

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]
    print(f"Train dataset size: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 4. Load Base Model
    print(f"Loading base model from: {config['model_params']['base_model_name_or_path']}")
    problem_type = "single_label_classification" if task_type == "classification" else None

    base_model = AutoModelForSequenceClassification.from_pretrained(
        config['model_params']['base_model_name_or_path'],
        num_labels=num_labels_for_model,
        problem_type=problem_type,
        trust_remote_code=config['model_params'].get('trust_remote_code', False)
    )
    if len(tokenizer) > base_model.config.vocab_size:  # Resize embeddings if tokenizer vocab is larger
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model token embeddings to: {len(tokenizer)}")
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # 5. Configure and Apply LoRA
    lora_config_params = config['lora_params']
    lora_peft_config = LoraConfig(
        r=lora_config_params['r'],
        lora_alpha=lora_config_params['lora_alpha'],
        target_modules=lora_config_params['target_modules'],
        lora_dropout=lora_config_params['lora_dropout'],
        bias=lora_config_params.get('bias', "none"),
        task_type=TaskType.SEQ_CLS  # LoRA task type for sequence classification/regression
    )
    try:
        lora_model = get_peft_model(base_model, lora_peft_config)
        print("LoRA model created successfully.")
        lora_model.print_trainable_parameters()
    except Exception as e:
        print(f"Error creating LoRA model: {e}")
        exit()
    lora_model = lora_model.to(DEVICE)

    # 6. Training Arguments
    train_args_config = config['training_params']
    metric_to_optimize = train_args_config['metric_for_best_model']

    # Determine greater_is_better for metric optimization
    if task_type == "regression":
        greater_is_better_val = metric_to_optimize in ['r2', 'spearmanr', 'pearsonr']
    else:  # Classification metrics like auc, f1, accuracy are typically higher is better
        greater_is_better_val = True
    greater_is_better_val = train_args_config.get('greater_is_better', greater_is_better_val)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_args_config['num_epochs'],
        per_device_train_batch_size=train_args_config['batch_size'],
        per_device_eval_batch_size=train_args_config.get('eval_batch_size', train_args_config['batch_size'] * 2),
        gradient_accumulation_steps=train_args_config.get('gradient_accumulation_steps', 1),
        learning_rate=train_args_config['learning_rate'],
        weight_decay=train_args_config.get('weight_decay', 0.01),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=train_args_config.get('logging_steps', 50),
        evaluation_strategy=train_args_config.get('evaluation_strategy', "epoch"),
        save_strategy=train_args_config.get('save_strategy', "epoch"),
        save_total_limit=train_args_config.get('save_total_limit', 2),
        load_best_model_at_end=True,
        metric_for_best_model=metric_to_optimize,
        greater_is_better=greater_is_better_val,
        fp16=train_args_config.get('fp16', torch.cuda.is_available()),
        seed=train_args_config.get('seed', 42),
        report_to=train_args_config.get("report_to", "none"),
        dataloader_num_workers=train_args_config.get('dataloader_num_workers', 0),
    )

    # 7. Trainer Initialization
    compute_metrics_fn = compute_metrics_classification if task_type == "classification" else compute_metrics_regression

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics_fn(p, config),
        # Pass config to metrics function for task_params access
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=train_args_config.get('early_stopping_patience', 3),
            early_stopping_threshold=train_args_config.get('early_stopping_threshold', 0.0)
        )] if train_args_config.get('use_early_stopping', True) else None,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    # 8. Training
    print("Starting LoRA fine-tuning...")
    train_output = trainer.train()
    trainer.log_metrics("train", train_output.metrics)
    trainer.save_metrics("train", train_output.metrics)
    print("Training finished.")

    best_adapter_path = os.path.join(output_dir, "best_lora_adapter")
    lora_model.save_pretrained(best_adapter_path)
    tokenizer.save_pretrained(best_adapter_path)
    print(f"Best LoRA adapter saved to {best_adapter_path}")

    # 9. Evaluation on Test Set
    print("\nEvaluating on the test set with the best LoRA model...")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    print(f"Test set metrics: {test_results}")

    # 10. Generate Predictions and Plots for Test Set
    print("\nGenerating predictions for Test set plots...")
    predictions_output = trainer.predict(test_dataset)
    raw_predictions = predictions_output.predictions
    true_labels = predictions_output.label_ids
    plot_output_dir = config['output_params'].get('plot_output_dir', os.path.join(output_dir, "plots"))
    os.makedirs(plot_output_dir, exist_ok=True)

    if task_type == "classification":
        if num_labels_for_model == 1:  # Binary classification (single logit)
            probabilities_for_plot = torch.sigmoid(torch.from_numpy(raw_predictions)).numpy().squeeze()
        else:  # Binary (2 logits) or Multiclass
            probabilities_all = torch.softmax(torch.from_numpy(raw_predictions), dim=-1).numpy()
            probabilities_for_plot = probabilities_all[:,
                                     1] if num_labels_for_model == 2 else probabilities_all  # For multiclass, pass all probabilities
        plot_classification_results(true_labels, probabilities_for_plot, plot_output_dir, config, prefix="test")
    elif task_type == "regression":
        plot_regression_results(true_labels, raw_predictions.squeeze(), plot_output_dir, config, prefix="test")

    print(f"\nAll plots and metrics saved to: {output_dir}")
    print("Unified LoRA fine-tuning script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified LoRA Fine-tuning Script for Classification and Regression using Hugging Face Datasets.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()
    main(args.config)
