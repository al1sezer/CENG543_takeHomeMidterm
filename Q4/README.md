# Question 4: Retrieval-Augmented Generation (RAG) System

## Overview
This notebook implements a complete **RAG (Retrieval-Augmented Generation)** pipeline for question answering, comparing **sparse retrieval (BM25)** vs **dense retrieval (SBERT)** methods combined with **FLAN-T5** generation.

## Dataset
- **HotpotQA**: Multi-hop reasoning question answering
- **Configuration**: Distractor (includes irrelevant documents)
- **Split**: Validation (7,405 examples)
- **Corpus**: ~52,000 unique documents extracted from contexts
- **Evaluation Set**: 200 random samples (seed=42)

## Notebook Structure

The notebook contains 7 sections with markdown explanations before each code cell:

| Section | Description | Key Output |
|---------|-------------|------------|
| 1. Environment Setup | Install dependencies | All packages ready |
| 2. Data Loading | Load HotpotQA, build corpus | 52K documents |
| 3. BM25 Retrieval | Sparse keyword-based retrieval | BM25 index |
| 4. SBERT Retrieval | Dense semantic retrieval | 384-dim embeddings |
| 5. RAG Pipeline | FLAN-T5 + retrieve_and_generate() | Working pipeline |
| 6. Evaluation | Precision, Recall, BLEU, ROUGE, BERTScore | 3 PNG plots |
| 7. Qualitative | Faithful vs Hallucination examples | 6 analyzed examples |

## System Components

### Retrieval Methods
1. **BM25 (Sparse)**: Traditional keyword-based retrieval
2. **SBERT (Dense)**: Neural semantic search using `all-MiniLM-L6-v2`

### Generator
- **FLAN-T5-Base**: 250M parameter instruction-tuned model
- **Context Window**: 512 tokens
- **Decoding**: Greedy + beam search (num_beams=5)

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. System Requirements
- **GPU**: Highly recommended for SBERT encoding (~2-5 min) and FLAN-T5 inference
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~2GB for models + dataset cache
- **Time**: ~60-90 minutes total (including corpus encoding)

## Reproducing Results

### Step 1: Run All Cells Sequentially
Open `midterm_q4.ipynb` in Jupyter/VS Code and execute cells in order:

1. **Section 1 - Environment Setup** (Markdown + Code): Installs all required packages
2. **Section 2 - Data Loading** (Markdown + Code): Loads HotpotQA validation split and builds corpus (~52K unique documents)
3. **Section 3 - BM25 Setup** (Markdown + Code): 
   - Tokenizes corpus with whitespace splitting
   - Creates BM25 index (Okapi BM25)
   - Tests retrieval with sample query
4. **Section 4 - SBERT Setup** (Markdown + Code): 
   - Loads `all-MiniLM-L6-v2` model (384-dim embeddings)
   - Encodes entire corpus (**Takes 2-5 minutes on GPU**)
   - Creates dense vector index with cosine similarity
5. **Section 5 - FLAN-T5 + RAG Pipeline** (Markdown + Code):
   - Loads FLAN-T5-Base model (250M parameters)
   - Implements `retrieve_and_generate()` function
   - Tests both BM25 and Dense retrieval with sample questions
6. **Section 6 - Comparative Evaluation** (Markdown + Code):
   - Evaluates both methods on 200 test samples (seed=42)
   - Retrieval Metrics: Precision@k, Recall@k (k=1,3,5)
   - Generation Metrics: BLEU-1, ROUGE-L, BERTScore
   - Generates 3 comparison visualizations (saved as PNG)
7. **Section 7 - Qualitative Analysis** (Markdown + Code):
   - Identifies 3 faithful generation examples (ROUGE-L > 0.4)
   - Identifies 3 hallucination examples (ROUGE-L < 0.15)
   - Analyzes retrieval vs generation failure modes

### Step 2: Verify Output Files
After running, check for:
- `retrieval_metrics_comparison.png` (Precision@k and Recall@k comparison)
- `generation_metrics_comparison.png` (BLEU, ROUGE-L, BERTScore comparison)
- `rag_complete_evaluation.png` (Complete evaluation - all metrics)
- Console output with 3 faithful + 3 hallucination examples

## Expected Results

### Quantitative Performance (200 test samples)
| Method | Precision@1 | Precision@3 | Precision@5 | Recall@1 | Recall@3 | Recall@5 | BLEU-1 | ROUGE-L | BERTScore |
|--------|-------------|-------------|-------------|----------|----------|----------|--------|---------|-----------|
| BM25 (Sparse) | ~0.35-0.45 | ~0.25-0.35 | ~0.20-0.30 | ~0.25-0.35 | ~0.40-0.50 | ~0.50-0.60 | ~0.12-0.18 | ~0.20-0.30 | ~0.80-0.85 |
| SBERT (Dense) | ~0.40-0.50 | ~0.30-0.40 | ~0.25-0.35 | ~0.30-0.40 | ~0.45-0.55 | ~0.55-0.65 | ~0.15-0.22 | ~0.25-0.35 | ~0.82-0.87 |

### Qualitative Examples

**Faithful Generation** (High ROUGE, answer in context):
```
Question: "Who are the vocals of Linkin Park?"
Context: "...Chester Bennington and Mike Shinoda..."
Generated: "Chester Bennington"
Status: ✅ Faithful (correctly extracted from context)
```

**Hallucination** (Low ROUGE, answer NOT in context):
```
Question: "When was the Eiffel Tower built?"
Context: "...The Louvre Museum in Paris..." (wrong document)
Generated: "1889" (from model's parametric knowledge)
Status: ❌ Hallucination (correct answer but not grounded in retrieved context)
```


## Hyperparameters

### Retrieval
```python
TOP_K = 5                    # Number of documents to retrieve
TEST_SAMPLES = 200           # Number of evaluation samples
RANDOM_SEED = 42             # For reproducibility
```

### BM25
```python
# No hyperparameters (default Okapi BM25)
```

### SBERT
```python
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384          # Model output dimension
DEVICE = 'cuda'              # GPU for encoding
SIMILARITY = 'cosine'        # Cosine similarity for retrieval
```

### FLAN-T5 Generation
```python
MODEL_NAME = 'google/flan-t5-base'  # 250M parameters
MAX_LENGTH = 1024                   # Max input tokens (with truncation)
MAX_NEW_TOKENS = 50                 # Max answer length
NUM_BEAMS = 5                       # Beam search width
EARLY_STOPPING = True               # Stop when confident
```

## Key Implementation Details

### Corpus Construction
```python
# From HotpotQA context structure
for row in dataset:
    context = row['context']  # List of [title, sentences]
    for title, sentences in zip(context['title'], context['sentences']):
        if title not in seen_titles:
            text = " ".join(sentences)
            corpus.append(text)
            doc_ids.append(title)
            seen_titles.add(title)
```
- Extracts all unique Wikipedia paragraphs from HotpotQA contexts
- Deduplicates by title (each title = one document)
- Total: ~52,000 documents

### BM25 Indexing
- Tokenization: Simple whitespace splitting
- Ranking: Okapi BM25 (term frequency + inverse document frequency)
- Speed: Milliseconds for retrieval

### SBERT Encoding
```python
# One-time encoding (expensive)
corpus_embeddings = sbert_model.encode(corpus, convert_to_tensor=True)

# Fast retrieval (cheap)
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)
```

### RAG Pipeline Flow
```
1. Question → Retriever → Top-K Documents
2. Documents → Context String (concatenate)
3. Prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
4. Prompt → FLAN-T5 → Generated Answer
```

### Context Window Management
- FLAN-T5-Base: 512 token limit
- TOP_K=5 provides balance (more context vs exceeding limit)
- Truncation enabled to handle overflow

## Evaluation Metrics

### Recall@3 (Retrieval Quality)
- Does the ground truth answer appear in top-3 retrieved documents?
- Range: 0-1 (higher is better)
- Critical bottleneck: if retrieval fails, generation will fail

### BLEU-1 (Generation Quality)
- Unigram precision between generated and reference answers
- Range: 0-1 (higher is better)

### ROUGE-L (Generation Quality)
- Longest Common Subsequence F1-score
- Range: 0-1 (higher is better)

### BERTScore (Semantic Similarity)
- Neural metric using BERT embeddings
- Range: 0-1 (higher is better)
- Captures paraphrases better than BLEU/ROUGE


## Failure Modes

### 1. Retrieval Bottleneck
- **Problem**: Relevant document not in top-K
- **Solution**: Increase K, use hybrid retrieval (BM25 + SBERT)

### 2. Context Window Overflow
- **Problem**: Too many documents exceed 512 tokens
- **Solution**: Reduce K, use reranking, or use larger model (T5-Large)

### 3. Multi-Hop Reasoning
- **Problem**: Answer requires combining info from multiple documents
- **Solution**: Iterative retrieval or chain-of-thought prompting

### 4. Parametric Knowledge Leakage
- **Problem**: Model answers from training data instead of context
- **Solution**: Stronger prompt engineering or instruction fine-tuning


```