<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/a7264691-db29-48e5-9f53-516e928ad653" />

# Resume–Job Matching with Sentence Transformers


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
2. **Resume-level split** (stronger evaluation; prevents leakage)

Metrics include:
- Accuracy
- Precision / Recall / F1
- ROC–AUC
- Optimal threshold for best F1
- Top-K accuracy (ranking multiple jobs per resume)

---

### Results — Resume-level split (recommended)
- **Accuracy:** 0.833
- **Precision:** 0.985
- **Recall:** 0.676
- **F1-score:** 0.802
- **ROC–AUC:** 0.929

<img width="662" height="501" alt="Screenshot 2025-08-10 at 10 45 13 PM" src="https://github.com/user-attachments/assets/64040399-9919-4852-953d-152995faa335" />

<img width="665" height="515" alt="Screenshot 2025-08-10 at 10 45 40 PM" src="https://github.com/user-attachments/assets/2873d006-c719-4bfe-a4cc-6763fa356a3a" />

<img width="627" height="504" alt="Screenshot 2025-08-10 at 10 46 28 PM" src="https://github.com/user-attachments/assets/936b8197-c3d5-480b-ac3a-3aeeae69d7b5" />

---

### Results — Pair-level split (for comparison)
- **Accuracy:** 0.957
- **Precision:** 0.992
- **Recall:** 0.922
- **F1-score:** 0.956
- **ROC–AUC:** 0.993

<img width="737" height="565" alt="image" src="https://github.com/user-attachments/assets/c06a4e10-2725-4b91-98a5-087d6d06e648" />

<img width="666" height="508" alt="Screenshot 2025-08-10 at 10 40 32 PM" src="https://github.com/user-attachments/assets/a3510601-cc85-4b82-941a-27e21df93939" />

<img width="629" height="507" alt="Screenshot 2025-08-10 at 10 40 59 PM" src="https://github.com/user-attachments/assets/6eaf436a-c6ad-443d-a73d-db5c2304ef44" />

---

## t-SNE Visualization
To understand category separation in embedding space, we visualized resume embeddings with **t-SNE**.
<img width="732" height="649" alt="Screenshot 2025-08-10 at 10 39 22 PM" src="https://github.com/user-attachments/assets/22684003-62c3-41e4-bc25-1b5d95f91730" />


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
