# Question 2: Neural Machine Translation with Attention Mechanisms

## Overview
This notebook compares three attention mechanisms for sequence-to-sequence neural machine translation: **Bahdanau (Additive)**, **Luong (Multiplicative)**, and **Scaled Dot-Product** attention.

## Dataset
- **Multi30k**: German → English translation
- **Splits**: Train (29,000), Validation (1,014), Test (1,000)
- **Source**: Hugging Face `bentrevett/multi30k`
- **Task**: Machine translation with multi-hop reasoning

## Models Evaluated
1. **Model 1**: Bidirectional GRU Encoder + Bahdanau Attention + GRU Decoder
2. **Model 2**: Bidirectional GRU Encoder + Luong Attention + GRU Decoder
3. **Model 3**: Bidirectional GRU Encoder + Scaled Dot-Product Attention + GRU Decoder

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# Download spaCy language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Reproducing Results

### Step 1: Run All Cells Sequentially
Open `midterm_q2.ipynb` in Jupyter/VS Code and execute cells in order:

1. **Setup** (Cell 1): Initializes seed=1234, device configuration
2. **Data Loading** (Cell 2): Loads Multi30k dataset via Hugging Face
3. **Tokenization** (Cell 3): spaCy tokenizers (German/English)
4. **Vocabulary Building** (Cell 4): Creates word-to-index mappings (MIN_FREQ=2)
5. **DataLoader** (Cell 5): Batching with padding (BATCH_SIZE=64)
6. **Encoder** (Cell 6): Bidirectional GRU encoder
7. **Bahdanau Attention** (Cell 7): Additive attention mechanism
8. **Luong Attention** (Cell 8): Multiplicative attention mechanism
9. **Scaled Dot-Product Attention** (Cell 9): Transformer-style attention
10. **Decoder** (Cell 10): GRU decoder with attention integration
11. **Seq2Seq Model** (Cell 11): Full encoder-decoder with bridge layer
12. **Training Functions** (Cell 12): Teacher forcing, loss computation
13. **Model 1 Training** (Cell 13): Bahdanau attention (10 epochs)
14. **Model 2 Training** (Cell 14): Luong attention (10 epochs)
15. **Model 2 Retry** (Cell 15): Re-trains Model 2 with different seed if needed
16. **Model 3 Training** (Cell 16): Scaled Dot-Product attention (10 epochs)
17. **Translation Utilities** (Cell 17): Greedy decoding function
18. **Final Evaluation** (Cell 18): BLEU-4 and ROUGE-L metrics on test set

### Step 2: Verify Output Files
After running, check for:
- `model-1-bahdanau-gru.pt` (Model 1 checkpoint)
- `model-2-luong-gru.pt` (Model 2 checkpoint)
- `model-3-scaled-dot-gru.pt` (Model 3 checkpoint)
- `attention_comparison_results.json` (Metrics summary)

## Expected Results

### Test Set Performance (BLEU-4 / ROUGE-L)
| Model | Attention Type | Test BLEU-4 | Test ROUGE-L | Test PPL |
|-------|----------------|-------------|--------------|--------------|
| Model 1 | Bahdanau (Additive) | 0.2850 | 0.5889 | 26.76 |
| Model 2 | Luong (Multiplicative) | 0.2438 | 0.5489 | 30.49 |
| Model 3 | Scaled Dot-Product | 0.2820 | 0.5797 | 29.66 |

**Key Findings**:
- Bahdanau achieves BEST performance across all metrics
- Luong underperforms due to over-sharp attention (near-zero entropy)
- Scaled Dot-Product shows competitive results, close to Bahdanau
- Higher attention entropy correlates with better translation quality
- Teacher forcing ratio=0.5 provides good balance

## Training Time (Approximate)
- **Per Model**: ~20-30 minutes per model (GPU), ~2-3 hours (CPU)
- **Total Runtime**: ~60-90 minutes for all 3 models (GPU)
- **Epochs**: 10 epochs per model (can be reduced to 5 for faster experimentation)

## Hyperparameters
```python
SEED = 1234
ENC_EMB_DIM = 256      # Encoder embedding dimension
DEC_EMB_DIM = 256      # Decoder embedding dimension
HID_DIM = 256          # GRU hidden dimension (per direction)
ENC_HID_DIM_CALC = 512 # Bidirectional encoder output (256 * 2)
DEC_HID_DIM_CALC = 256 # Decoder hidden dimension
ATTN_DIM = 256         # Attention dimension (Bahdanau)
N_LAYERS = 2           # Number of GRU layers
ENC_DROPOUT = 0.5      # Encoder dropout
DEC_DROPOUT = 0.5      # Decoder dropout
BATCH_SIZE = 64        # Training batch size
N_EPOCHS = 10          # Training epochs
LEARNING_RATE = 0.001  # Adam optimizer
CLIP = 1               # Gradient clipping
TEACHER_FORCING = 0.5  # Teacher forcing ratio
```

## Key Implementation Details

### Source Reversal
- **Status**: NOT applied in this implementation
- **Reason**: Bidirectional GRU encoder captures both forward and backward context
- **Note**: Source reversal is optional for bidirectional encoders

### Attention Mechanisms

**1. Bahdanau (Additive)**:
```
score(h_t, h_s) = v^T tanh(W_1 h_t + W_2 h_s)
```
- Learns alignment via feed-forward network
- Original attention mechanism (2015)

**2. Luong (Multiplicative)**:
```
score(h_t, h_s) = h_t^T W h_s
```
- Simpler, faster than Bahdanau
- Used in Google's NMT system

**3. Scaled Dot-Product**:
```
score(h_t, h_s) = (h_t^T h_s) / sqrt(d)
```
- Transformer-style attention
- Scaling prevents gradient vanishing in deep models

### Vocabulary Construction
- **MIN_FREQ**: 2 (words must appear at least twice)
- **Special Tokens**: `<unk>`, `<pad>`, `<sos>`, `<eos>`
- **German Vocab**: ~7,800 tokens
- **English Vocab**: ~5,900 tokens


## Evaluation Metrics

### BLEU-4 (BiLingual Evaluation Understudy)
- Measures n-gram precision (1-4 grams)
- Range: 0-100 (higher is better)
- Standard metric for machine translation

### ROUGE-L (Longest Common Subsequence)
- Measures F1-score of longest matching subsequence
- Range: 0-1 (higher is better)
- Focuses on structural similarity



