"""
Medical Summarization Fine-tuning with Professional Logging and Wandb Integration

Features:
- Professional logging with Rich console + file logging
- Epoch-based training with per-epoch evaluation
- Custom evaluation callback for wandb metrics logging
- Per-epoch inference with sample predictions logged to wandb
- Best checkpoint tracking and management

Usage:
    python summarization/fine_tune_llama.py --config config.yml
    python summarization/fine_tune_llama.py --config config.yml --quick_test
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
from pathlib import Path
import time
from datetime import datetime
import yaml
import json
import shutil

# Load environment variables from .env file
load_dotenv()

# Import professional logger
from utils.logger import get_logger, get_console, log_separator, log_dict, log_metrics

# Initialize logger for this module
logger = get_logger(__name__)
console = get_console()

# Rich library for progress and tables
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback
from datasets import load_dataset
from peft import (
    get_peft_model, 
    prepare_model_for_kbit_training, 
    LoraConfig,
    PeftModel
)
from trl import SFTTrainer, SFTConfig
import evaluate
from rouge_score import rouge_scorer
import wandb

transformers.logging.set_verbosity_info()


class EvaluationCallback(TrainerCallback):
    """
    Custom callback for comprehensive evaluation and wandb logging.
    
    This callback:
    - Computes ROUGE and other metrics at the end of each epoch
    - Generates sample predictions for qualitative monitoring
    - Logs metrics and samples to wandb
    - Tracks and saves the best checkpoint
    """
    
    def __init__(self, eval_dataset, tokenizer, generate_prompt_fn, config, 
                 inference_samples=None, output_dir=None):
        """
        Initialize the evaluation callback.
        
        Args:
            eval_dataset: Validation dataset
            tokenizer: Model tokenizer
            generate_prompt_fn: Function to generate prompts
            config: Configuration object
            inference_samples: List of samples for per-epoch inference
            output_dir: Output directory for checkpoints
        """
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.generate_prompt = generate_prompt_fn
        self.config = config
        self.inference_samples = inference_samples or []
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize metrics
        self.bertscore = evaluate.load("bertscore")
        
        logger.info(f"EvaluationCallback initialized with {len(self.inference_samples)} inference samples")
        self.initial_eval_done = False
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl,
                       model=None, **kwargs):
        """Run initial evaluation at epoch 0 (before any training)."""
        if self.initial_eval_done:
            return
            
        logger.info("Running initial evaluation (epoch 0) before training...")
        
        if model is None:
            logger.warning("No model available for initial evaluation")
            return
        
        # Generate sample predictions at step 0
        if self.inference_samples and self.config.log_samples_to_wandb:
            self._generate_and_log_samples(model, step=0, state=state)
        
        self.initial_eval_done = True
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl,
                    model=None, metrics=None, **kwargs):
        """Called after every evaluation (every eval_steps)."""
        current_step = state.global_step
        epoch = state.epoch if state.epoch else 0
        
        logger.info(f"Evaluation at step {current_step} (epoch {epoch:.2f})")
        
        # Log metrics
        if metrics:
            log_metrics(logger, metrics)
        
        # Skip if no model available
        if model is None:
            logger.warning("No model available for evaluation callback")
            return
        
        # Generate sample predictions
        if self.inference_samples and self.config.log_samples_to_wandb:
            self._generate_and_log_samples(model, step=current_step, state=state)
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, 
                     model=None, **kwargs):
        """Called at the end of each epoch - just log epoch completion."""
        epoch = int(state.epoch)
        logger.info(f"Epoch {epoch} completed.")
    
    def _generate_and_log_samples(self, model, step, state):
        """Generate predictions on sample examples and log to wandb."""
        logger.info(f"Generating {len(self.inference_samples)} sample predictions at step {step}...")
        
        predictions = []
        model.eval()
        
        with torch.no_grad():
            for i, sample in enumerate(self.inference_samples):
                input_prompt = self.generate_prompt(sample["text"])
                input_tokens = self.tokenizer(
                    input_prompt, 
                    return_tensors="pt"
                )["input_ids"].to(self.device)
                
                try:
                    with torch.cuda.amp.autocast():
                        generation_output = model.generate(
                            input_ids=input_tokens,
                            max_new_tokens=self.config.max_new_tokens,
                            eos_token_id=self.tokenizer.eos_token_id,
                            do_sample=self.config.do_sample,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    
                    prediction = self.tokenizer.decode(
                        generation_output[0], 
                        skip_special_tokens=True
                    )
                    prediction = prediction[len(input_prompt):].strip()
                except Exception as e:
                    logger.error(f"Error generating prediction for sample {i}: {e}")
                    prediction = "[GENERATION ERROR]"
                
                predictions.append({
                    "step": step,
                    "sample_id": i,
                    "source": sample["text"][:500] + "..." if len(sample["text"]) > 500 else sample["text"],
                    "reference": sample["summary"],
                    "prediction": prediction
                })
                
                # Log individual sample
                logger.info(f"Sample {i+1} prediction generated (len={len(prediction)})")
        
        # Log to wandb as table
        if self.config.use_wandb and wandb.run is not None:
            try:
                table = wandb.Table(
                    columns=["step", "sample_id", "source", "reference", "prediction"],
                    data=[[p["step"], p["sample_id"], p["source"], 
                           p["reference"], p["prediction"]] for p in predictions]
                )
                wandb.log({f"predictions/step_{step}": table})
                logger.info(f"Logged {len(predictions)} sample predictions to wandb at step {step}")
            except Exception as e:
                logger.error(f"Error logging predictions to wandb: {e}")
        
        # Save predictions to file
        predictions_file = self.output_dir / f"predictions_step_{step}.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved predictions to {predictions_file}")
        
        model.train()


class BestCheckpointCallback(TrainerCallback):
    """
    Callback to track and maintain the best checkpoint.
    
    Creates a symlink or copy at 'best_checkpoint/' directory.
    """
    
    def __init__(self, output_dir, metric_name="eval_loss", greater_is_better=False):
        self.output_dir = Path(output_dir)
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = float('-inf') if greater_is_better else float('inf')
        self.best_checkpoint_path = None
        
        logger.info(f"BestCheckpointCallback tracking {metric_name} (greater_is_better={greater_is_better})")
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, 
                    metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics is None:
            return
        
        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            logger.warning(f"Metric {self.metric_name} not found in evaluation metrics")
            return
        
        is_better = (current_metric > self.best_metric if self.greater_is_better 
                     else current_metric < self.best_metric)
        
        if is_better:
            self.best_metric = current_metric
            epoch = int(state.epoch) if state.epoch else 0
            
            # Find the latest checkpoint
            checkpoints = list(self.output_dir.glob("checkpoint-*"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                self.best_checkpoint_path = latest_checkpoint
                
                # Create best_checkpoint directory
                best_dir = self.output_dir / "best_checkpoint"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(latest_checkpoint, best_dir)
                
                logger.info(f"New best {self.metric_name}: {current_metric:.4f} at epoch {epoch}")
                logger.info(f"Best checkpoint saved to {best_dir}")
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.run.summary[f"best_{self.metric_name}"] = current_metric
                    wandb.run.summary["best_epoch"] = epoch


class RichProgressCallback(TrainerCallback):
    """Custom callback for Rich progress bar and logging"""
    
    def __init__(self, num_epochs, logging_steps):
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.progress = None
        self.task_id = None
        self.start_time = None
        self.current_epoch = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize progress tracking at training start"""
        self.start_time = time.time()
        logger.info("Training started")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        self.current_epoch = int(state.epoch) + 1 if state.epoch else 1
        logger.info(f"Starting epoch {self.current_epoch}/{self.num_epochs}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        elapsed = time.time() - self.start_time
        logger.info(f"Epoch {self.current_epoch} completed. Total time: {elapsed/60:.2f} min")
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to console with Rich formatting"""
        if logs:
            # Filter out non-essential logs
            important_keys = ['loss', 'eval_loss', 'learning_rate', 'epoch', 
                            'rouge1', 'rouge2', 'rougeL', 'bertscore']
            filtered_logs = {k: v for k, v in logs.items() 
                           if any(key in k.lower() for key in important_keys)}
            
            if filtered_logs:
                log_metrics(logger, filtered_logs)
            
    def on_train_end(self, args, state, control, **kwargs):
        """Clean up at training end"""
        elapsed_time = time.time() - self.start_time
        logger.info(f"Training completed! Total time: {elapsed_time/60:.2f} minutes")


def load_config(config_path="config.yml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Medical summarization fine-tuning with config file support."
    )
    parser.add_argument(
        "--config", type=str, default="config.yml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="Run quick test with minimal examples"
    )
    
    # Optional overrides for config values
    parser.add_argument("--model_name_or_path", type=str, help="Override model path")
    parser.add_argument("--output_path", type=str, help="Override output path")
    parser.add_argument("--num_train_epochs", type=int, help="Override epochs")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    
    args = parser.parse_args()
    
    # Load config file
    config = load_config(args.config)
    
    # Apply quick test settings if enabled
    if args.quick_test:
        logger.info("Quick test mode enabled!")
        config['quick_test']['enabled'] = True
    
    # Create a namespace object with all config values
    class ConfigNamespace:
        pass
    
    cfg = ConfigNamespace()
    
    # Model settings
    cfg.model_name_or_path = args.model_name_or_path or config['model']['name_or_path']
    cfg.load_in_8bit = config['model']['load_in_8bit'] and torch.cuda.is_available()
    cfg.trust_remote_code = config['model']['trust_remote_code']
    cfg.device_map = config['model']['device_map']
    cfg.device = config['system']['device']
    
    # LoRA settings
    cfg.lora_rank = config['lora']['rank']
    cfg.lora_alpha = config['lora']['alpha']
    cfg.lora_dropout = config['lora']['dropout']
    cfg.num_target_modules = config['lora']['num_target_modules']
    
    # Training settings (use quick test if enabled, otherwise use normal config)
    if config['quick_test']['enabled']:
        cfg.num_train_epochs = config['quick_test']['num_train_epochs']
        cfg.logging_steps = config['quick_test']['logging_steps']
        cfg.num_train_examples = config['quick_test']['num_train_examples']
        cfg.num_val_examples = config['quick_test']['num_val_examples']
        cfg.num_test_examples = config['quick_test']['num_test_examples']
    else:
        cfg.num_train_epochs = args.num_train_epochs or config['training']['num_train_epochs']
        cfg.logging_steps = config['training']['logging_steps']
        cfg.num_train_examples = config['data']['num_train_examples']
        cfg.num_val_examples = config['data']['num_val_examples']
        cfg.num_test_examples = config['data']['num_test_examples']
    
    cfg.batch_size = config['training']['per_device_train_batch_size']
    cfg.gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    cfg.learning_rate = float(args.learning_rate or config['training']['learning_rate'])
    cfg.lr_scheduler_type = config['training']['lr_scheduler_type']
    cfg.warmup_ratio = float(config['training']['warmup_ratio'])
    cfg.max_grad_norm = float(config['training']['max_grad_norm'])
    cfg.optim = config['training']['optim']
    cfg.eval_strategy = config['training']['evaluation_strategy']
    cfg.eval_steps = config['training'].get('eval_steps', 500)
    cfg.save_strategy = config['training']['save_strategy']
    cfg.save_steps = config['training'].get('save_steps', 500)
    cfg.save_total_limit = config['training']['save_total_limit']
    cfg.load_best_model_at_end = config['training']['load_best_model_at_end']
    cfg.metric_for_best_model = config['training']['metric_for_best_model']
    cfg.greater_is_better = config['training']['greater_is_better']
    
    # Data settings
    cfg.data_path = config['data']['base_path']
    cfg.train_file = config['data']['train_file']
    cfg.validation_file = config['data']['validation_file']
    cfg.test_file = config['data']['test_file']
    cfg.max_source_length = config['data']['max_source_length']
    cfg.max_target_length = config['data']['max_target_length']
    
    # Prompt settings
    cfg.instruction_text = config['prompt']['instruction']
    cfg.response_text = config['prompt']['response_prefix']
    
    # Output settings
    cfg.output_path = args.output_path or config['training']['output_dir']
    
    # Evaluation settings
    cfg.evaluation = config['evaluation']['evaluation_only']
    cfg.evaluation_model_path = config['evaluation']['evaluation_model_path']
    
    # Logging settings
    cfg.use_wandb = config['logging']['use_wandb']
    cfg.wandb_project = config['logging']['wandb_project']
    cfg.use_rich = config['logging']['use_rich']
    
    # Inference settings
    cfg.num_eval_samples = config['inference']['num_eval_samples']
    cfg.max_new_tokens = config['inference']['max_new_tokens']
    cfg.do_sample = config['inference']['do_sample']
    cfg.log_samples_to_wandb = config['inference']['log_samples_to_wandb']
    
    # Early stopping settings
    cfg.early_stopping_enabled = config['training'].get('early_stopping', {}).get('enabled', False)
    cfg.early_stopping_patience = config['training'].get('early_stopping', {}).get('patience', 3)
    cfg.early_stopping_threshold = config['training'].get('early_stopping', {}).get('threshold', 0.001)
    
    return cfg


def get_rouge_score(gold, pred):
    """Compute ROUGE scores for a single prediction."""
    rouge_scores = ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouge_scores, use_stemmer=True)
    scores = scorer.score(gold, pred)
    return {k: scores[k].fmeasure * 100 for k in rouge_scores}


def compute_custom_metrics(srcs, golds, preds, device):
    """Compute comprehensive evaluation metrics."""
    scores = defaultdict(list)
    bertscore = evaluate.load("bertscore")
    sari = evaluate.load("sari")
    
    # Compute ROUGE and length for each example
    for gold, pred in zip(golds, preds):
        for k, v in get_rouge_score(gold, pred).items():
            scores[k].append(v)
        scores['words'].append(len(pred.split(' ')))
    
    for k, v in scores.items():
        scores[k] = np.mean(v)

    # BERTScore
    scores['bert_score'] = np.mean(
        bertscore.compute(predictions=preds, references=golds, lang="en", device=device)['f1']
    ) * 100
    scores['bert_score_deberta'] = np.mean(
        bertscore.compute(
            predictions=preds, references=golds, device=device, 
            model_type="microsoft/deberta-large-mnli"
        )['f1']
    ) * 100
    
    # SARI
    scores['sari'] = sari.compute(
        sources=srcs, predictions=preds, references=[[g] for g in golds]
    )['sari']

    return scores


def print_metrics_as_latex(metrics):
    """Print metrics as LaTeX table row."""
    order = ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 
             'bert_score', 'bert_score_deberta', 'sari', 'words']
    print(' & '.join([f'${metrics[k]:.2f}$' for k in order]))


def main():
    """Main training function."""
    args = parse_args()
    output_dir = args.output_path
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Log training configuration
    log_separator(logger, "BHC Medical Summarization Fine-tuning")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info(f"Data path: {args.data_path}")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # LoRA configuration
    target_modules = {
        1: ['q_proj'], 
        2: ['q_proj', 'v_proj'], 
        4: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    }
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules[args.num_target_modules],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Log configuration
    log_dict(logger, {
        "LoRA Rank": args.lora_rank,
        "LoRA Alpha": args.lora_alpha,
        "LoRA Dropout": args.lora_dropout,
        "Target Modules": target_modules[args.num_target_modules],
        "Learning Rate": args.learning_rate,
        "Epochs": args.num_train_epochs,
        "Batch Size": args.batch_size,
        "Gradient Accumulation": args.gradient_accumulation_steps,
    }, title="Training Configuration")

    # Initialize wandb
    if args.use_wandb:
        short_model_name = args.model_name_or_path.split('/')[-1]
        project_name = f"{args.wandb_project}_{short_model_name}"
        if args.evaluation:
            project_name += "_evaluation"
        else:
            project_name += "_finetuning"
        
        wandb.init(
            project=project_name,
            entity=None,
            config=vars(args),
            tags=["medical-summarization", "qwen" if "qwen" in args.model_name_or_path.lower() else "llm"],
            notes=f"Fine-tuning {args.model_name_or_path} on MIMIC-IV BHC"
        )
        logger.info(f"Wandb initialized: {project_name}")
    else:
        logger.warning("Wandb logging disabled")
        
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    hf_token = os.environ.get('HF_TOKEN', '')
    model_name = args.model_name_or_path
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=hf_token, 
        trust_remote_code=args.trust_remote_code
    )
    
    logger.info("Model and tokenizer loaded successfully!")
    
    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Load data
    data_path = args.data_path
    data_files = {
        "train": os.path.join(data_path, args.train_file),
        "validation": os.path.join(data_path, args.validation_file),
        "test": os.path.join(data_path, args.test_file),
    }
    
    logger.info("Loading datasets...")
    for split, path in data_files.items():
        logger.info(f"  {split}: {path}")
    
    data = load_dataset("json", data_files=data_files)
    data_train, data_test, data_val = data["train"], data["test"], data["validation"]

    # Limit number of examples
    if args.num_train_examples:
        data_train = data_train.select(range(0, len(data_train))[-args.num_train_examples:])
    if args.num_val_examples:
        data_val = data_val.select(range(0, len(data_val))[-args.num_val_examples:])
    if args.num_test_examples:
        data_test = data_test.select(range(0, len(data_test))[-args.num_test_examples:])
    
    log_dict(logger, {
        "Training": len(data_train),
        "Validation": len(data_val),
        "Test": len(data_test),
    }, title="Dataset Statistics")

    # Prompt generation function
    instruction_text = args.instruction_text
    response_text = args.response_text

    def generate_prompt(reference, summary=None, eos_token=None):
        if eos_token is None:
            eos_token = tokenizer.eos_token
            
        instruction = f"{instruction_text}\n"
        input_text = f"{reference}\n"
        response = f"{response_text} {summary + ' ' + eos_token if summary else ''} "
        return ''.join([instruction, input_text, response])
    
    def truncate_text(example, tokens=None):
        if tokens is None:
            tokens = args.max_source_length
        example['truncated'] = False
        while tokenizer(
            generate_prompt(example["text"], example["summary"]), 
            return_tensors="pt"
        )["input_ids"].shape[1] >= tokens:
            example["text"] = example["text"].rsplit('.', 1)[0]
            example['truncated'] = True
        return example
        
    # Truncate texts
    logger.info("Truncating texts to fit context window...")
    data_train = data_train.map(truncate_text)
    data_val = data_val.map(truncate_text)
    data_test = data_test.map(truncate_text)
    
    log_dict(logger, {
        "Train truncated": f"{sum(data_train['truncated'])}/{len(data_train)}",
        "Val truncated": f"{sum(data_val['truncated'])}/{len(data_val)}",
        "Test truncated": f"{sum(data_test['truncated'])}/{len(data_test)}",
    }, title="Truncation Summary")

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    if args.evaluation:
        logger.info(f"Loading model for evaluation: {args.evaluation_model_path}")
        trained_model = PeftModel.from_pretrained(
            model, args.evaluation_model_path, torch_dtype=torch.float16
        ).to(device)
    else:
        # Training mode
        if args.load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        # Training arguments
        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=args.batch_size, 
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_eval_batch_size=args.batch_size,
            eval_accumulation_steps=args.gradient_accumulation_steps,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            num_train_epochs=args.num_train_epochs,
            logging_steps=args.logging_steps,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=args.greater_is_better,
            optim=args.optim,
            lr_scheduler_type=args.lr_scheduler_type,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            group_by_length=True,
            ddp_find_unused_parameters=False,
            report_to="wandb" if args.use_wandb else "none",
            max_length=args.max_source_length,
            bf16=False,
            fp16=False,
            use_cpu=not torch.cuda.is_available(),
            eval_on_start=True,  # Run evaluation before training starts
        )

        # Preprocess dataset
        def preprocess_dataset(dataset):
            def add_text_column(example):
                example["text"] = generate_prompt(example["text"], example["summary"])
                return example
            return dataset.map(add_text_column)
        
        logger.info("Preprocessing datasets...")
        data_train_processed = preprocess_dataset(data_train)
        data_val_processed = preprocess_dataset(data_val)

        # Select inference samples
        num_samples = min(args.num_eval_samples, len(data_val))
        inference_samples = [data_val[i] for i in range(num_samples)]
        logger.info(f"Selected {num_samples} samples for per-epoch inference")

        # Create callbacks
        callbacks = []
        
        # Rich progress callback
        if args.use_rich:
            callbacks.append(RichProgressCallback(args.num_train_epochs, args.logging_steps))
        
        # Evaluation callback for wandb metrics and inference
        eval_callback = EvaluationCallback(
            eval_dataset=data_val,
            tokenizer=tokenizer,
            generate_prompt_fn=generate_prompt,
            config=args,
            inference_samples=inference_samples,
            output_dir=output_dir,
        )
        callbacks.append(eval_callback)
        
        # Best checkpoint callback
        best_ckpt_callback = BestCheckpointCallback(
            output_dir=output_dir,
            metric_name=args.metric_for_best_model,
            greater_is_better=args.greater_is_better,
        )
        callbacks.append(best_ckpt_callback)
        
        # Early stopping callback (if enabled)
        if args.early_stopping_enabled:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold
            )
            callbacks.append(early_stopping_callback)
            logger.info(f"Early stopping enabled: patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold}")
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=data_train_processed,
            eval_dataset=data_val_processed,
            peft_config=lora_config,
            processing_class=tokenizer,
            callbacks=callbacks,
            args=training_args,
        )

        # Upcast layer norms for stable training
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)

        log_separator(logger, "Starting Training")
        trainer.train()
        
        # Save best model
        logger.info("Saving final model...")
        trainer.save_model(f"{output_dir}/best_val_loss")
        logger.info(f"Model saved to {output_dir}/best_val_loss")
        trained_model = trainer.model
    
    # Generate predictions on test set
    log_separator(logger, "Generating Test Predictions")
    
    predictions_test = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Predicting...", total=len(data_test))
        
        for ex in data_test:
            input_prompt = generate_prompt(ex["text"])
            input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to(device)
            
            with torch.cuda.amp.autocast():
                generation_output = trained_model.generate(
                    input_ids=input_tokens,
                    max_new_tokens=350,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prediction = tokenizer.decode(generation_output[0], skip_special_tokens=True)
            prediction = prediction[len(input_prompt):].strip()
            predictions_test.append(prediction)
            progress.advance(task)
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics_test = compute_custom_metrics(
        data_test["text"], data_test["summary"], predictions_test, device
    )
    
    # Log sample predictions
    logger.info("Sample predictions:")
    for i in range(min(len(data_test), 3)):
        logger.info(f"--- Example {i+1} ---")
        logger.info(f"Reference: {data_test[i]['summary'][:200]}...")
        logger.info(f"Prediction: {predictions_test[i][:200]}...")
        
    # Save predictions
    output_path = Path(output_dir)
    with open(output_path / 'predictions_test.jsonl', "w") as f:
        f.write("\n".join(predictions_test))

    with open(output_path / 'predictions_test_dict.jsonl', "w") as f:
        for pred in predictions_test:
            f.write(f'{{"summary": "{pred}"}}\n')

    # Log final metrics
    log_separator(logger, "Final Test Metrics")
    log_metrics(logger, metrics_test)
    
    # Round and log to wandb
    metrics_test_rounded = {k: round(v, 2) for k, v in metrics_test.items()}
    
    if args.use_wandb:
        wandb.log({"test/" + k: v for k, v in metrics_test_rounded.items()})
        logger.info("Metrics logged to wandb")
    
    logger.info("LaTeX format:")
    print_metrics_as_latex(metrics_test_rounded)
    
    # Final summary
    log_separator(logger, "Training Complete")
    logger.info(f"Model saved to: {output_dir}/best_val_loss")
    logger.info(f"Predictions saved to: {output_path / 'predictions_test.jsonl'}")
    logger.info(f"Best ROUGE-1: {metrics_test_rounded.get('rouge1', 0):.2f}")
    logger.info(f"Best BERTScore: {metrics_test_rounded.get('bert_score', 0):.2f}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
