#!/usr/bin/env python3
"""
QLoRA fine-tuning for Qwen2.5-0.5B with full CLI, TensorBoard, and train/val/test.

Default corpus: drive/MyDrive/aicompliance/corpus.jsonl (mount Drive first on Colab)

Usage:
  python scripts/finetune_qwen_experiment.py
  python scripts/finetune_qwen_experiment.py --lora_r 32 --learning_rate 5e-5
  python scripts/finetune_qwen_experiment.py --run_name exp_lr5e5 --learning_rate 5e-5

TensorBoard:
  tensorboard --logdir outputs/qwen-0.5b-aicompliance/runs
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# -----------------------------------------------------------------------------
# Defaults (golden standard)
# -----------------------------------------------------------------------------
DEFAULT_CORPUS = "drive/MyDrive/aicompliance/corpus.jsonl"
DEFAULT_OUTPUT_BASE = "drive/MyDrive/aicompliance/outputs"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TEXT_COLUMN = "text"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_EXAMPLES_FOR_SPLIT = 30


def parse_args():
    p = argparse.ArgumentParser(
        description="Qwen QLoRA fine-tuning with train/val/test and TensorBoard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Run
    p.add_argument("--run_name", type=str, default=None,
                    help="Experiment name (default: timestamp)")
    p.add_argument("--corpus", type=str, default=DEFAULT_CORPUS,
                    help="Path to corpus.jsonl")
    p.add_argument("--output_base", type=str, default=DEFAULT_OUTPUT_BASE,
                    help="Base dir for outputs and TensorBoard runs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha (scale)")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout")
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,v_proj,k_proj,o_proj",
                   help="Comma-separated LoRA target modules")

    # Quantization
    p.add_argument("--load_in_4bit", action="store_true", default=True,
                   help="Use 4-bit quantization (QLoRA)")
    p.add_argument("--no_load_in_4bit", action="store_false", dest="load_in_4bit")
    p.add_argument("--use_double_quant", action="store_true", default=True,
                   help="Double quantization for 4-bit")
    p.add_argument("--no_use_double_quant", action="store_false", dest="use_double_quant")
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                   choices=["nf4", "fp4"],
                   help="4-bit quant type")
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"],
                   help="Compute dtype for 4-bit")

    # Training
    p.add_argument("--learning_rate", type=float, default=2e-5,
                   help="Peak learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="Weight decay")
    p.add_argument("--num_train_epochs", type=int, default=3,
                   help="Number of epochs")
    p.add_argument("--per_device_train_batch_size", type=int, default=4,
                   help="Train batch size per device")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--warmup_steps", type=float, default=200,
                   help="Warmup steps")
    p.add_argument("--max_seq_length", type=int, default=512,
                   help="Max sequence length")
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Max gradient norm for clipping")

    # Eval / logging
    p.add_argument("--eval_strategy", type=str, default="steps",
                   choices=["no", "steps", "epoch"],
                   help="When to run validation")
    p.add_argument("--eval_steps", type=int, default=50,
                   help="Eval every N steps")
    p.add_argument("--save_steps", type=int, default=50,
                   help="Save checkpoint every N steps")
    p.add_argument("--logging_steps", type=int, default=10,
                   help="Log every N steps")
    p.add_argument("--save_total_limit", type=int, default=2,
                   help="Max checkpoints to keep")
    p.add_argument("--load_best_model_at_end", action="store_true", default=True,
                   help="Load best checkpoint by eval_loss at end")

    args = p.parse_args()
    args.lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    return args


def main():
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_base, "qwen-0.5b-aicompliance", run_name)
    tb_log_dir = os.path.join(args.output_base, "qwen-0.5b-aicompliance", "runs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Save run config for reproducibility
    config_path = os.path.join(output_dir, "run_config.json")
    config_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {config_path}")

    # --- Dataset ---
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=args.corpus, split="train")
    n_total = len(dataset)
    print(f"Total examples: {n_total}")

    if n_total >= MIN_EXAMPLES_FOR_SPLIT:
        tmp = dataset.train_test_split(test_size=(1 - TRAIN_RATIO), seed=args.seed)
        train_ds = tmp["train"]
        holdout = tmp["test"]
        n_holdout = len(holdout)
        val_size = int(n_holdout * VAL_RATIO / (VAL_RATIO + TEST_RATIO))
        val_test = holdout.train_test_split(test_size=(n_holdout - val_size), seed=args.seed)
        val_ds = val_test["train"]
        test_ds = val_test["test"]
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    else:
        train_ds = dataset
        val_ds = test_ds = None
        print("Too few examples for val/test split.")

    # --- Model + LoRA ---
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=args.use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Trainer ---
    from trl import SFTConfig, SFTTrainer

    current_run_logdir = os.path.join(tb_log_dir, run_name)
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy if val_ds else "no",
        eval_steps=args.eval_steps if val_ds else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end and val_ds is not None,
        metric_for_best_model="eval_loss" if val_ds else None,
        report_to="tensorboard",
        run_name=run_name,
        remove_unused_columns=False,
        dataset_text_field=TEXT_COLUMN,
        max_length=args.max_seq_length,
        packing=False,
    )
    if hasattr(sft_config, "logging_dir"):
        sft_config.logging_dir = current_run_logdir

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Train
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Test set evaluation
    if test_ds is not None:
        from datasets import Dataset

        def tokenize_for_eval(examples):
            tok = tokenizer(
                examples[TEXT_COLUMN],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            tok["labels"] = [[t if t != tokenizer.pad_token_id else -100 for t in seq] for seq in tok["input_ids"]]
            return tok

        test_tokenized = test_ds.map(
            tokenize_for_eval,
            batched=True,
            remove_columns=test_ds.column_names,
            desc="Tokenize test set",
        )
        test_metrics = trainer.evaluate(eval_dataset=test_tokenized)
        test_path = os.path.join(output_dir, "test_metrics.json")
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        print("=== Test set metrics ===")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"Test metrics saved to {test_path}")

    print(f"\nCompare runs: tensorboard --logdir {tb_log_dir}")


if __name__ == "__main__":
    main()
