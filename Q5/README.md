# Question 5: Interpretability & Diagnostic Evaluation

## 📋 Overview

This module implements **interpretability and diagnostic evaluation** for a German-to-English Neural Machine Translation (NMT) Transformer model. The goal is to understand model behavior through saliency maps, uncertainty estimation, and failure case analysis.

## 🎯 Objectives

| Task | Description |
|------|-------------|
| **Task A** | Load pre-trained Transformer model from Q3 |
| **Task B** | Compute **Saliency Maps** (Input Importance via Gradients) |
| **Task C** | Identify and analyze **Failure Cases** |
| **Task D** | Measure **Uncertainty** (Entropy-based confidence) |

## 📁 Project Structure

```
Q5/
├── midterm_q5.ipynb           # Main notebook with all analysis
├── transformer_model_final.pt  # Pre-trained model checkpoint (from Q3)
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## 🔧 Requirements

### Dependencies
```
torch>=2.0.0
transformers>=4.30.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
```

### Hardware
- **GPU**: CUDA-enabled GPU recommended (works on CPU but slower)
- **RAM**: Minimum 8GB recommended

## 🚀 Reproducibility Guide

### Step 1: Environment Setup
```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Model Checkpoint
Ensure `transformer_model_final.pt` is in the Q5 folder. This file contains:
- `model_state_dict`: Trained model weights
- `epoch`: Training epoch number
- `optimizer_state_dict`: Optimizer state
- `loss`: Final loss value

The checkpoint is loaded from Q3's trained Transformer model.

### Step 3: Run Notebook
Execute cells in order:
1. **Section 1**: Imports & Model Architecture definitions
2. **Section 2**: Load DistilBERT tokenizers, encoder, and pre-trained Transformer model
3. **Section 3**: Analysis pipeline function (`analyze_prediction`)
4. **Section 4**: Visualization helper (`plot_analysis`)
5. **Section 5**: Execute diagnostic evaluation with 6 test cases

## ⚙️ Model Configuration

### Transformer Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `HID_DIM` | 256 | Hidden/Embedding dimension |
| `ENC_LAYERS` | 2 | Number of encoder layers |
| `DEC_LAYERS` | 2 | Number of decoder layers |
| `HEADS` | 8 | Number of attention heads |
| `FF_DIM` | 512 | Feed-forward dimension |
| `DROPOUT` | 0.1 | Dropout rate |

### Pre-trained Models Used
| Model | Purpose |
|-------|---------|
| `distilbert-base-german-cased` | Source (German) tokenizer & feature extraction |
| `distilbert-base-uncased` | Target (English) tokenizer |
| `transformer_model_final.pt` | Trained NMT model from Q3 |

## 📊 Analysis Pipeline

### 1. Input Processing
```
German Text → DistilBERT → 768-dim vectors → Transformer → English Text
```

### 2. Saliency Map Computation
```python
# Gradient-based input importance
score = last_token_logits.max()
score.backward()
saliency = src_vectors.grad.norm(dim=2)  # L2 norm per token
```

### 3. Uncertainty Estimation
```python
# Entropy-based confidence
probs = F.softmax(logits, dim=0)
entropy = -torch.sum(probs * torch.log(probs + 1e-9))
# Higher entropy = More uncertain
```

## 🧪 Test Cases

Six test cases for diagnostic evaluation (1 success baseline + 5 failure cases):

| Case Type | German Input | Expected Output | Purpose |
|-----------|--------------|-----------------|---------|
| **Success (Baseline)** | "Ein Hund läuft." | "A dog is running." | Simple Multi30k-style sentence |
| **Failure 1: Ambiguity** | "Das Schloss ist sehr alt." | "The castle/lock is very old." | Word sense disambiguation (Schloss: castle vs lock) |
| **Failure 2: Negation** | "Die Kinder spielen nicht im Garten." | "The children are not playing in the garden." | Negation word "nicht" handling |
| **Failure 3: Idiom** | "Das ist nicht mein Bier." | "That's not my problem." | Idiomatic expression (literal: "That's not my beer") |
| **Failure 4: Rare Words (OOV)** | "Der Wissenschaftler analysiert die Molekularstruktur." | "The scientist analyzes the molecular structure." | Technical/scientific vocabulary |
| **Failure 5: Long Dependency** | "Der Mann, der den Hund hat, der braun ist, geht." | "The man who has the dog that is brown is walking." | Nested relative clauses |

## 📈 Output Visualizations

### Saliency Map (Input Importance)
- Bar chart showing which input tokens most influenced the translation
- Higher bars = more important tokens

### Uncertainty Plot (Entropy per Step)
- Line chart showing model confidence at each decoding step
- Higher entropy = less confident prediction

### Summary Table
- DataFrame with all test cases, predictions, and average entropy

## 🔄 Checkpoint Format

The `transformer_model_final.pt` file is saved as a dictionary:
```python
{
    'epoch': int,                    # Training epoch
    'model_state_dict': dict,        # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'loss': float                    # Final loss value
}
```

Loading method:
```python
checkpoint = torch.load('transformer_model_final.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
```



