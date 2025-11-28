# Question 3: Transition from Recurrence to Self-Attention

## 📋 Overview

This project analyzes the **empirical and conceptual transition** from recurrent neural networks (RNN) to self-attention-driven architectures (Transformer) for neural machine translation. The implementation uses the **Multi30k German-English** dataset and compares both architectures under fair experimental conditions.

## 🎯 Objectives

- Re-implement a Seq2Seq model with **Bahdanau (Additive) Attention**
- Train a **Transformer** model on the identical dataset
- Incorporate **DistilBERT embeddings** for consistent feature extraction
- Evaluate models using **BLEU, ROUGE-L, training stability, and computational efficiency**
- Conduct **ablation studies** on layer depth and attention heads
- Analyze how self-attention enables **global dependency modeling** and **parallelization**

## 🏗️ Architecture

### 1. RNN Seq2Seq with Bahdanau Attention

```
┌─────────────────────────────────────────────────────────────┐
│                        ENCODER                               │
│  [768-dim BERT Vectors] → Linear(768→256) → Bidirectional GRU│
│  Output: [Batch, Seq, 512] + Hidden States                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   BAHDANAU ATTENTION                         │
│  energy = tanh(W[hidden; encoder_outputs])                   │
│  attention = softmax(V × energy)                             │
│  context = Σ(attention × encoder_outputs)                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        DECODER                               │
│  GRU with attention-weighted context vectors                 │
│  Output: [Batch, Seq, Vocab_Size]                            │
└─────────────────────────────────────────────────────────────┘
```

### 2. Transformer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ENCODER                                  │
│  [768-dim BERT] → Linear(768→256) → Positional Encoding     │
│  → N × (Multi-Head Self-Attention + FFN + LayerNorm)        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     DECODER                                  │
│  Token Embedding → Positional Encoding                       │
│  → N × (Masked Self-Attention + Cross-Attention + FFN)      │
│  → Linear(256 → Vocab_Size)                                  │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Q3/
├── midterm_q3.ipynb          # Main notebook with all experiments
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── rnn_model_final.pt     # Trained RNN checkpoint
├── transformer_model_final.pt  # Trained Transformer checkpoint
├── ablation_models/       # Ablation study checkpoints
│   ├── transformer_light.pt   # 1 layer, 4 heads
│   └── transformer_deep.pt    # 4 layers, 8 heads
└── *.png                  # Generated visualization plots
```

## ⚙️ Hyperparameters

| Parameter | RNN | Transformer | Notes |
|-----------|-----|-------------|-------|
| Hidden Dimension | 256 | 256 | Same for fair comparison |
| Encoder Layers | 2 | 2 | Bidirectional for RNN |
| Decoder Layers | 2 | 2 | - |
| Attention Heads | - | 8 | Multi-head attention |
| Feed-Forward Dim | - | 512 | 2× hidden dim |
| Dropout | 0.1 | 0.1 | Same for both |
| Learning Rate | 0.0005 | 0.0005 | Adam optimizer |
| Batch Size | 64 | 64 | - |
| Epochs | 5 | 5 | - |

## 🚀 Installation

```bash
# Clone or navigate to the project directory
cd Q3

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (HuggingFace)
- datasets
- sacrebleu
- rouge-score
- matplotlib
- tqdm
- numpy

## 📊 Evaluation Metrics

### Translation Quality
- **BLEU Score**: N-gram precision with brevity penalty
- **ROUGE-L**: Longest common subsequence F-measure

### Computational Efficiency
- **Inference Time**: Total time to translate test set
- **GPU Memory**: Peak memory usage during inference
- **Parameter Count**: Total trainable parameters

### Training Stability
- **Loss Variance**: Standard deviation of training loss
- **Generalization Gap**: Difference between validation and training loss

## 🔬 Ablation Study

Three Transformer configurations are tested:

| Config | Layers | Heads | Purpose |
|--------|--------|-------|---------|
| **Light** | 1 | 4 | Speed optimization, fewer parameters |
| **Base** | 2 | 8 | Standard configuration |
| **Deep** | 4 | 8 | Higher capacity, deeper learning |

### Attention Mechanism

**Bahdanau (Additive) Attention:**
```
score(s_t, h_i) = v^T × tanh(W_a[s_t; h_i])
```

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

## 🖼️ Generated Visualizations


1. `training_stability_comparisonreal.png` - Perplexity trends
2. `training_Stability_metrics.png` - Stability bar charts
3. `ablation_stabilitypng.png` - Ablation PPL trends
4. `final_evaluation_metrics.png` - BLEU, ROUGE, Time, Memory

## 🔧 Usage

### Running the Notebook

1. Open `midterm_q3.ipynb` in Jupyter/Colab/VS Code
2. Run cells sequentially (1.1 → 6.1)
3. Models will be saved automatically after training

### Loading Trained Models

```python
import torch

# Load RNN Model
checkpoint = torch.load('rnn_model_final.pt')
model_rnn.load_state_dict(checkpoint['model_state_dict'])

# Load Transformer Model
checkpoint = torch.load('transformer_model_final.pt')
model_trans.load_state_dict(checkpoint['model_state_dict'])
```
