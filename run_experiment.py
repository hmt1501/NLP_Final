import os
import subprocess
import sys

# Set encoding and disable wandb
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["WANDB_DISABLED"] = "true"

# Define command
cmd = [
    r"D:\miniconda\envs\all\python.exe",
    "summarization/run_summarization.py",
    "--model_name_or_path", "allenai/led-large-16384",
    "--do_train",
    "--do_eval",
    "--do_predict",
    "--train_file", "dataset/mimic-iv-note-ext-di-bhc/dataset/train_4000_600_chars.json",
    "--validation_file", "dataset/mimic-iv-note-ext-di-bhc/dataset/valid_4000_600_chars.json",
    "--test_file", "dataset/mimic-iv-note-ext-di-bhc/dataset/test_4000_600_chars_last_100.json",
    "--output_dir", "models/led-large-16384/mimic-iv-note-di-bhc_led-large-16384_4000_600_chars_100_valid/dropout_0.05_learning_rate_5e-4",
    "--max_steps", "200000",
    "--eval_strategy", "steps",
    "--eval_steps", "20000",
    "--save_steps", "20000",
    "--load_best_model_at_end",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--dropout", "0.05",
    "--learning_rate", "5e-4",
    "--predict_with_generate",
    "--max_source_length", "4096",
    "--max_target_length", "350"
]

# Run command
subprocess.run(cmd)
