# Resume–Job Matching with Sentence Transformers

### One Sentence Summary
A fine-tuned sentence transformer model that matches candidate resumes to job descriptions using semantic similarity, evaluated with rigorous metrics and visualizations.

---

## Overview
This project uses **Sentence Transformers** to encode resumes and job descriptions into dense embeddings, then computes **cosine similarity** to measure match quality. By fine-tuning the `all-MiniLM-L6-v2` model on a labeled dataset of resumes and their job categories, the model learns domain-specific relationships beyond keyword matching.

Unlike traditional keyword-based search, this approach captures **semantic meaning**, enabling robust matching even if the job description and resume use different vocabulary.

---

## Summary of Work Done

### Data
- **Source:** Labeled resume dataset with 25+ job categories (e.g., IT, Finance, HR, Healthcare).
- **Size:** ~6,000–7,000 resumes after cleaning.
- **Labels:** Each resume assigned a single job category.

**Preprocessing Steps:**
- Removed HTML tags, non-ASCII characters, and noise.
- Balanced categories and tokenized text.
- Created positive/negative pairs for training.

---

### Problem Formulation

**Input:**
- Resume text
- Job description text

**Output:**
- Similarity score between 0 and 1, indicating match strength.

---

## Models & Training

- **Base Model:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **Loss:** CosineSimilarityLoss
- **Batch size:** 16
- **Epochs:** 1 (demo-level; can be extended)
- **Optimizer:** AdamW
- **Hardware:** Google Colab T4 GPU

**Training Duration:** ~1 minute

---

## Evaluation

Two evaluation setups were used:
1. **Pair-level split** (original approach)
2. **Resume-level split** (stronger evaluation; prevents leakage) ✅

Metrics include:
- Accuracy
- Precision / Recall / F1
- ROC–AUC
- Optimal threshold for best F1
- Top-K accuracy (ranking multiple jobs per resume)

---

### Results — Resume-level split (recommended)
- **Accuracy:** TBD
- **Precision:** TBD
- **Recall:** TBD
- **F1-score:** TBD
- **ROC–AUC:** TBD
- **Best F1 threshold:** TBD (F1=TBD)
- **Top-3 Accuracy:** TBD

**Figures:**
![ROC (resume-level)](figures/roc_resume_level.png)  <!-- 📌 Save your ROC plot to this path -->
![Score Distribution (resume-level)](figures/score_distribution_resume_level.png)  <!-- 📌 Save your histogram to this path -->
![Confusion Matrix (resume-level)](figures/confusion_matrix_resume_level.png)  <!-- 📌 Already saved by helper cell -->

---

### Results — Pair-level split (for comparison)
- **Accuracy:** TBD
- **Precision:** TBD
- **Recall:** TBD
- **F1-score:** TBD
- **ROC–AUC:** TBD
- **Best F1 threshold:** TBD (F1=TBD)

**Figures:**
![ROC (pair-level)](figures/roc_pair_level.png)
![Score Distribution (pair-level)](figures/score_distribution_pair_level.png)
![Confusion Matrix (pair-level)](figures/confusion_matrix_pair_level.png)

---

## t-SNE Visualization
To understand category separation in embedding space, we visualized resume embeddings with **t-SNE** (legend maps category index → name).

![t-SNE](figures/tsne_resume_embeddings.png)  <!-- 📌 Save your t-SNE figure here -->

---

## Key Takeaways
- **Semantic matching** beats keyword matching for resume–job relevance.
- Resume-level split ensures realistic evaluation without data leakage.
- Visualizations make results interpretable for recruiters and developers.

---

## Future Work
- Fine-tune for more epochs and include **hard negatives**.
- Try larger models like `mpnet-base-v2`.
- Build a **Streamlit app** for real-time matching and ranking.
- Add LLM explanations to justify match scores and suggest resume improvements.

---

## How to Reproduce
1. **Install:**
```bash
pip install sentence-transformers pandas scikit-learn matplotlib
```
2. **Run:** Open `Resume_Matching_UPGRADED.ipynb` in Colab and execute all cells.
3. **Dataset:** Point to your resume dataset or use the Kaggle dataset.
4. **Outputs:** Plots are saved in `figures/` when you run the saving helper cell.

---
