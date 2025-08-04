<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/577441c1-5aa1-4fb2-a7a4-17936a2fae62" />

# Resume-Job Matching with Sentence Transformers

This repository presents a semantic similarity-based NLP solution for matching candidate resumes to job descriptions using sentence embeddings from transformer models.

---

##  Overview

The goal is to measure how well a candidate's resume aligns with a job description by computing cosine similarity between their text embeddings. This approach enables robust, language-aware matching without relying on strict keyword overlap.

We fine-tuned the `all-MiniLM-L6-v2` sentence transformer model on a dataset of categorized resumes to improve its understanding of domain-specific relationships. Once trained, the model was used to encode both resumes and job descriptions into dense vectors, and similarity scores were calculated.

---

##  Summary of Work Done

###  Data

- **Type**: Text-based resume dataset
- **Inputs**: Resume content and their labeled job categories
- **Labels**: 25+ job role categories (e.g., IT, Finance, HR, Healthcare)
- **Size**: ~6,000–7,000 entries post-cleaning

###  Preprocessing

- Cleaned HTML and stringified content from resume files
- Verified class balance across job categories
- Tokenized and encoded input data
- Converted resume pairs into `(text1, text2, label)` format for supervised fine-tuning

---

##  Data Visualization

1. **Resume Category Distribution**  
   Showed fairly even distribution across job roles like IT, Business Development, Engineering, etc.

2. **Model Training Logs**  
   Used `SentenceTransformer`'s built-in training loop with cosine similarity loss.

3. **Similarity Score Example Output**  
   A sample score of `0.7944` indicated strong alignment between a resume and a job description.

---

##  Problem Formulation

- **Input**:  
  Two pieces of text — a candidate resume and a job description

- **Output**:  
  A **similarity score** between 0 and 1, indicating how well the resume matches the job.

### Model

- **Base Model**: `all-MiniLM-L6-v2`
- **Fine-tuned** using:
  - **Loss**: `CosineSimilarityLoss`
  - **Batch size**: 16
  - **Epochs**: 1 (demo-level)
  - **Optimizer**: AdamW (default)

---

##  Training

- **Environment**:
  - Google Colab (GPU: T4)
  - Python 3, Jupyter Notebook
  - Libraries: `sentence-transformers`, `pandas`, `torch`

- **Training Duration**: Under 1 minute on T4 GPU
- **Output**: Fine-tuned model saved to `/output/` directory

---

##  Evaluation

- **Metric**: Cosine similarity
- **Example**:
  ```plaintext
  Resume: "Experienced data analyst with skills in Python, SQL, and Tableau."
  Job: "Looking for a data analyst to create dashboards and analyze trends."
  Similarity score: 0.7944
  ```
- Higher scores correlate with greater semantic match.

---

##  Conclusions

- The sentence transformer model was effective for understanding resume-job relationships.
- Cosine similarity provided an interpretable, continuous measure of relevance.
- Model can generalize to various job domains due to pretrained embeddings.

---

##  Future Work

- Fine-tune for more epochs and include hard negative examples
- Use a larger model (e.g., `paraphrase-MiniLM`, `mpnet-base`)
- Build a web interface to allow recruiters to drop in job descriptions and get ranked resumes
- Experiment with generative LLMs (e.g., GPT-4) to **explain** match scores

---

##  How to Reproduce Results

1. Clone the repo and open the notebook:
   ```
   Resume_Job_Matching.ipynb
   ```

2. Run all cells to:
   - Load and prepare data
   - Fine-tune the model
   - Evaluate on sample resume-job pairs

3. You can tweak the resume/job description in the last cell to test your own examples.

---

##  Repository Contents

| File | Description |
|------|-------------|
| `Resume_Job_Matching.ipynb` | Main notebook with full pipeline |
| `data/` | Contains preprocessed and labeled resumes |
| `output/` | Fine-tuned model saved here |
| `README.md` | This file |

---

##  Citations

- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Models](https://huggingface.co/models)
- Resume dataset adapted from a publicly available Kaggle resume classification dataset.
