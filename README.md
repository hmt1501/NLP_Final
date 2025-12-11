# A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models

![Figure1-5](https://github.com/user-attachments/assets/fa631f08-9e56-4a37-aea3-3b46fd6d31ef)

This repository contains the code to reproduce the results of the paper [A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models](https://proceedings.mlr.press/v248/hegselmann24a.html) by Stefan Hegselmann, Shannon Zejiang Shen, Florian Gierse, Monica Agrawal, David Sontag, and Xiaoyi Jiang.

We released the 100 doctor-written summaries from the MIMIC-IV-Note Discharge Instructions and hallucinations 100 LLM-generated patient summaries annotated for unsupported facts by two medical experts on PhysioNet. We also published all datasets created in our work to fully reproduce our experiments.

If you consider our work helpful or use our datasets, please consider the citations for our paper and PhysioNet repository:

```bibtex
@InProceedings{pmlr-v248-hegselmann24a,
  title = \t{A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models},
  author =      {Hegselmann, Stefan and Shen, Zejiang and Gierse, Florian and Agrawal, Monica and Sontag, David and Jiang, Xiaoyi},
  booktitle = \t{Proceedings of the fifth Conference on Health, Inference, and Learning},
  pages = \t{339--379},
  year = \t{2024},
  volume = \t{248},
  series = \t{Proceedings of Machine Learning Research},
  month = \t{27--28 Jun},
  publisher =   {PMLR},
  url = \t{https://proceedings.mlr.press/v248/hegselmann24a.html},
}

@Misc{hegselmann_ann-pt-summ2024,
  title = \t{Medical Expert Annotations of Unsupported Facts in {Doctor}-{Written} and LLM-Generated Patient Summaries},
  author =      {Hegselmann, Stefan and Shen, Zejiang and Gierse, Florian and Agrawal, Monica and Sontag, David and Jiang, Xiaoyi},
  booktitle = \t{Proceedings of the fifth Conference on Health, Inference, and Learning},
  year = \t{2024},
  publisher =   {PhysioNet},
  url = \t{https://physionet.org/content/ann-pt-summ/1.0.0/},
  doi = \t{https://doi.org/10.13026/a66y-aa53},
}
```

---

## Table of Contents

- [Problem Description](#problem-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)

---

## Problem Description

### Task: Medical Document Summarization

**Objective:** Generate faithful and high-quality patient summaries from hospital discharge notes using Large Language Models.

**Input:**
- Clinical discharge notes from MIMIC-IV database
- Long-form medical text containing patient history, diagnosis, treatments, and care instructions
- Text length: Up to 4,000 characters (after preprocessing)
- Format: Unstructured clinical narratives with medical terminology

**Output:**
- Concise patient summaries suitable for discharge instructions
- Target length: Approximately 350-600 characters
- Focus: Key medical information, diagnosis, treatment plan, and follow-up instructions
- Requirement: Faithful to source (no hallucinations), medically accurate

**Challenges:**
1. **Long documents:** Hospital notes can be very lengthy, requiring models that handle long sequences
2. **Medical accuracy:** Summaries must be factually correct and supported by source text
3. **Information density:** Must preserve critical medical information while being concise
4. **Domain-specific language:** Requires understanding of medical terminology and context

**Evaluation Metrics:**
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for content overlap
- BERTScore for semantic similarity
- SARI for simplification quality
- Medical expert annotation for hallucination detection

---

## Model Architecture

### LED (Longformer Encoder-Decoder)

The primary model used in this repository is **LED-large-16384** (`allenai/led-large-16384`), specifically designed for long document summarization.

**Why LED for Medical Summarization?**

1. **Long Context Window:** 
   - Handles up to 16,384 tokens (vs 512 for standard BERT)
   - Critical for processing lengthy hospital discharge notes
   - Uses efficient attention mechanism (O(n) vs O(n²))

2. **Architecture Components:**
   ```
   Encoder (Longformer):
   - Local + Global Attention
   - Sliding window attention for local context
   - Global attention for important tokens
   - 16 layers, 1024 hidden size
   
   Decoder (Transformer):
   - Standard autoregressive decoder
   - 12 layers, 1024 hidden size
   - Cross-attention to encoder outputs
   
   Total Parameters: ~406M
   ```

3. **Attention Mechanism:**
   - **Local Attention:** Each token attends to k tokens on left and right (default window = 1024)
   - **Global Attention:** Special tokens attend to all other tokens
   - Enables efficient processing of long documents

4. **Configuration Used:**
   ```python
   max_source_length: 4096 tokens
   max_target_length: 350 tokens
   dropout: 0.05
   learning_rate: 5e-4
   batch_size: 1 (per device)
   max_steps: 200,000
   ```

**Model Inputs:**
- Tokenized discharge notes (up to 4096 tokens)
- Attention masks
- Decoder input IDs (for training)

**Model Outputs:**
- Generated summary tokens
- Attention weights (for interpretability)
- Loss values (during training)

**Alternative Models Supported:**
- Llama 2 variants (see `summarization/README.md`)
- GPT-4 (see `gpt-4/README.md`)

---

## Dataset

### MIMIC-IV-Note Discharge Instructions with Brief Hospital Course

**Source:** MIMIC-IV Clinical Database (requires PhysioNet credentialed access)

**Dataset Location:** `dataset/mimic-iv-note-ext-di-bhc/dataset/`

**Files:**
- `train_4000_600_chars.json` - Training set (20,931 examples)
- `valid_4000_600_chars.json` - Validation set (2,608 examples)
- `test_4000_600_chars.json` - Test set (2,608 examples)
- `test_4000_600_chars_last_100.json` - Small test subset (100 examples)

**Dataset Format (JSONL):**
```json
{
  "text": "Brief Hospital Course: Patient with history of...",
  "summary": "Patient was admitted for... Discharge instructions include..."
}
```

**Each Example Contains:**
- `text`: Source hospital discharge note
  - Preprocessed to ~4000 characters max
  - Includes: brief hospital course, diagnoses, medications, procedures
  - Contains medical terminology, abbreviations, clinical findings

- `summary`: Target patient summary
  - ~350-600 characters
  - Doctor-written discharge instructions
  - Simplified language for patient understanding

**Data Statistics:**
- **Training examples:** 20,931
- **Validation examples:** 2,608
- **Test examples:** 2,608
- **Average input length:** ~3,900 characters
- **Average summary length:** ~600 characters
- **Domain:** Internal medicine, surgery, cardiology, etc.

**Preprocessing:**
- Text truncation to 4000 characters
- Summary truncation to 600 characters
- Medical entity preservation
- Deidentification (MIMIC-IV is already deidentified)

**Dataset Access:**
1. Obtain MIMIC-IV access from PhysioNet (requires credentialing)
2. Follow preprocessing pipeline in `preprocess/` directory
3. Or use our published preprocessed datasets

---

## Setup

### 1. Environment Setup

**Requirements:**
- Python 3.11+
- CUDA-capable GPU recommended (training)
- 16GB+ RAM
- Windows/Linux/macOS

**Create conda environment:**
```bash
conda create -n all python=3.11
conda activate all
```

### 2. Install Dependencies

**Install PyTorch (CPU version shown, adjust for GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Install all requirements:**
```bash
pip install -r requirements.txt
```

**Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from transformers import AutoModelForSeq2SeqLM; print('Transformers OK')"
```

**Expected output:**
```
PyTorch: 2.9.1+cpu (or your version)
Transformers OK
```

---

## Training

### Quick Start - Train LED Model

**Full training command:**
```bash
conda activate all

python summarization/run_summarization.py \
  --model_name_or_path allenai/led-large-16384 \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file "dataset/mimic-iv-note-ext-di-bhc/dataset/train_4000_600_chars.json" \
  --validation_file "dataset/mimic-iv-note-ext-di-bhc/dataset/valid_4000_600_chars.json" \
  --test_file "dataset/mimic-iv-note-ext-di-bhc/dataset/test_4000_600_chars_last_100.json" \
  --output_dir "models/led-large-16384/my_experiment" \
  --max_steps 200000 \
  --eval_strategy steps \
  --eval_steps 20000 \
  --save_steps 20000 \
  --load_best_model_at_end \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --dropout 0.05 \
  --learning_rate 5e-4 \
  --predict_with_generate \
  --max_source_length 4096 \
  --max_target_length 350
```

### Training Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name_or_path` | `allenai/led-large-16384` | Pre-trained LED model |
| `max_steps` | `200000` | Total training steps |
| `eval_steps` | `20000` | Evaluate every 20k steps |
| `save_steps` | `20000` | Save checkpoint every 20k steps |
| `per_device_train_batch_size` | `1` | Batch size per GPU (large model) |
| `learning_rate` | `5e-4` | Learning rate |
| `dropout` | `0.05` | Dropout probability |
| `max_source_length` | `4096` | Max input tokens |
| `max_target_length` | `350` | Max output tokens |
| `predict_with_generate` | - | Use generation for evaluation |

### Training Time Estimates

**On single GPU (e.g., RTX 3090):**
- ~20-30 seconds per step
- 200,000 steps ≈ 45-60 days

**Recommendations:**
- Use multiple GPUs with distributed training
- Reduce `max_steps` for faster experimentation (e.g., 50,000)
- Use gradient accumulation for larger effective batch size
- Monitor with TensorBoard: `tensorboard --logdir models/`

### Training Outputs

**Directory structure:**
```
models/led-large-16384/my_experiment/
├── checkpoint-20000/          # Saved checkpoints
├── checkpoint-40000/
├── ...
├── test_generations.txt       # Test predictions (text)
├── test_generations.pkl       # Test predictions (pickle)
├── trainer_state.json         # Training state
├── train_results.json         # Training metrics
├── eval_results.json          # Validation metrics
└── test_results.json          # Test metrics
```

### Monitoring Training

**Weights & Biases (disabled by default):**
To enable wandb logging, comment out lines 308-309 in `summarization/run_summarization.py`:
```python
# os.environ["WANDB_DISABLED"] = "true"
# os.environ["WANDB_MODE"] = "disabled"
```

Then uncomment lines 323-326 to initialize wandb.

**TensorBoard:**
```bash
tensorboard --logdir models/ --port 6006
```

---

## Inference

### Generate Summaries for New Texts

**1. Using trained model:**

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # For Windows

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your trained model
model_path = "models/led-large-16384/my_experiment"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Input text (hospital discharge note)
hospital_note = """
Brief Hospital Course: Patient is a 65-year-old male with history of 
hypertension and diabetes who presented with chest pain. EKG showed 
ST-elevation in anterior leads. Patient underwent emergent cardiac 
catheterization with stent placement to LAD. Post-procedure course 
was uncomplicated. Patient was started on dual antiplatelet therapy.
"""

# Tokenize
inputs = tokenizer(
    hospital_note, 
    max_length=4096, 
    truncation=True, 
    return_tensors="pt"
)

# Generate summary
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=350,
    num_beams=4,
    early_stopping=True
)

# Decode
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated Summary:", summary)
```


**2. Batch inference from file:**

```python
import jsonlines

# Load test data
test_data = list(jsonlines.open('dataset/.../test_file.json'))

summaries = []
for example in test_data:
    inputs = tokenizer(example['text'], max_length=4096, 
                      truncation=True, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=350)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summaries.append(summary)
```

**3. Using the script for prediction only:**

```bash
python summarization/run_summarization.py \
  --model_name_or_path models/led-large-16384/my_experiment \
  --do_predict \
  --test_file "your_input_file.json" \
  --output_dir "predictions/" \
  --per_device_eval_batch_size 1 \
  --predict_with_generate \
  --max_source_length 4096 \
  --max_target_length 350
```

**Output:** Predictions saved to `predictions/test_generations.txt`

### Generation Parameters

```python
model.generate(
    input_ids,
    max_length=350,           # Maximum summary length
    min_length=50,            # Minimum summary length
    num_beams=4,              # Beam search width (higher = better quality, slower)
    length_penalty=2.0,       # Penalty for length (>1.0 favors longer)
    early_stopping=True,      # Stop when all beams find EOS
    no_repeat_ngram_size=3,   # Prevent repetition
    temperature=1.0,          # Sampling temperature
    top_k=50,                 # Top-k sampling
    top_p=0.95                # Nucleus sampling
)
```

---

## Project Structure

```
patient_summaries_with_llms/
├── dataset/                          # Datasets
│   └── mimic-iv-note-ext-di-bhc/
│       └── dataset/                  # Preprocessed JSONL files
├── summarization/                    # Main training code
│   ├── run_summarization.py         # Training script
│   └── README.md                    # Detailed instructions
├── gpt-4/                           # GPT-4 experiments
│   └── README.md
├── hallucination_detection/         # Hallucination detection
│   └── README.md
├── labeling/                        # MedTator labeling tools
│   └── README.md
├── notebooks/                       # Jupyter notebooks
│   └── README.md
├── preprocess/                      # Preprocessing pipeline
│   └── README.md
├── scripts/                         # Parameter tuning scripts
│   └── README.md
├── models/                          # Saved models (created during training)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Additional Resources

### Detailed Component Documentation

* [summarization/README.md](summarization/README.md): LED and Llama 2 training details
* [gpt-4/README.md](gpt-4/README.md): GPT-4 experiments
* [hallucination_detection/README.md](hallucination_detection/README.md): Hallucination detection methods
* [labeling/README.md](labeling/README.md): MedTator labeling analysis
* [notebooks/README.md](notebooks/README.md): Jupyter notebooks for analysis
* [preprocess/README.md](preprocess/README.md): Data preprocessing pipeline
* [scripts/README.md](scripts/README.md): Hyperparameter tuning scripts

### Troubleshooting

**Common Issues:**

1. **Out of Memory (OOM):**
   - Reduce `per_device_train_batch_size` to 1
   - Reduce `max_source_length` (e.g., 2048)
   - Use gradient checkpointing
   - Use mixed precision training (fp16)

2. **Slow Training:**
   - Use multiple GPUs with `torchrun`
   - Increase batch size with gradient accumulation
   - Use faster storage (SSD)

3. **Poor Summaries:**
   - Train longer (more steps)
   - Adjust learning rate
   - Try different generation parameters (beam size, length penalty)
   - Ensure adequate training data

4. **PyTorch Version Error:**
   - Ensure PyTorch >= 2.6 (see requirements.txt)
   - Use: `pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu`

### Citation

If you use this code or datasets, please cite our paper and PhysioNet repository (see top of README).

### License

This project is licensed under the terms specified in [LICENSE](LICENSE).

### Contact

For questions or issues, please open a GitHub issue or contact the authors.
