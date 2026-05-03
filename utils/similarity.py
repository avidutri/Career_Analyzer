"""
utils/similarity.py
--------------------
Recommendation engine: matches resume skills against job roles using:
  1. TF-IDF + Cosine Similarity  (primary, no GPU needed)
  2. Jaccard Similarity           (fast set-based fallback / comparison)

Also handles:
  - Skill gap analysis (missing skills, match %)
  - Top-N role recommendations with scores
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.skill_extraction import extract_skills

# ---------------------------------------------------------------------------
# Dataset loading & preprocessing
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "data" / "jobs_dataset.csv"


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """
    Load jobs dataset and normalise the required_skills column to a
    clean comma-separated string and a Python list.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise skills: lowercase, strip whitespace
    df["required_skills"] = df["required_skills"].apply(
        lambda s: ",".join(sk.strip().lower() for sk in str(s).split(","))
    )
    df["skills_list"] = df["required_skills"].apply(
        lambda s: [sk.strip() for sk in s.split(",") if sk.strip()]
    )
    return df


# ---------------------------------------------------------------------------
# TF-IDF model  (trained once, reused for inference)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent.parent / "models" / "tfidf_model.pkl"


def _skill_tokenizer(s: str) -> list[str]:
    """Module-level tokenizer so it can be pickled."""
    return [t.strip() for t in s.split(",")]


def train_tfidf(df: pd.DataFrame, save: bool = True) -> tuple[TfidfVectorizer, np.ndarray]:
    """
    Fit a TF-IDF vectorizer on job skill strings.
    Each job role is represented as a bag-of-skills document.

    Returns:
        vectorizer: Fitted TfidfVectorizer
        job_matrix: (n_jobs × vocab) sparse matrix
    """
    corpus = df["required_skills"].tolist()
    vectorizer = TfidfVectorizer(
        tokenizer=_skill_tokenizer,
        lowercase=True,
        token_pattern=None,        # we supply our own tokenizer
    )
    job_matrix = vectorizer.fit_transform(corpus)

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "job_matrix": job_matrix, "df": df}, f)
        print(f"[TF-IDF] Model saved to {MODEL_PATH}")

    return vectorizer, job_matrix


def load_tfidf() -> tuple[TfidfVectorizer, np.ndarray, pd.DataFrame] | None:
    """Load a previously saved TF-IDF model. Returns None if not found."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        return bundle["vectorizer"], bundle["job_matrix"], bundle["df"]
    return None


def get_or_train_tfidf() -> tuple[TfidfVectorizer, np.ndarray, pd.DataFrame]:
    """Load cached model or train a fresh one."""
    cached = load_tfidf()
    if cached:
        return cached
    df = load_dataset()
    vectorizer, job_matrix = train_tfidf(df, save=True)
    return vectorizer, job_matrix, df


# ---------------------------------------------------------------------------
# Skill gap analysis
# ---------------------------------------------------------------------------

def analyze_skill_gap(
    resume_skills: list[str],
    job_skills: list[str],
) -> dict:
    """
    Compare resume skills against a job's required skills.

    Returns a dict with:
        matched_skills   – skills the candidate already has
        missing_skills   – skills required but not in resume
        match_score      – (matched / required) × 100
    """
    resume_set = set(s.lower().strip() for s in resume_skills)
    job_set = set(s.lower().strip() for s in job_skills)

    matched = sorted(resume_set & job_set)
    missing = sorted(job_set - resume_set)
    score = round(len(matched) / len(job_set) * 100, 2) if job_set else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_score": score,
        "total_required": len(job_set),
        "total_matched": len(matched),
    }


# ---------------------------------------------------------------------------
# Cosine-similarity recommendation
# ---------------------------------------------------------------------------

def recommend_jobs_tfidf(
    resume_skills: list[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Recommend top-N job roles based on TF-IDF cosine similarity.

    Args:
        resume_skills: List of skills extracted from resume.
        top_n: Number of recommendations to return.

    Returns:
        List of dicts sorted by similarity score (descending).
    """
    vectorizer, job_matrix, df = get_or_train_tfidf()

    # Represent resume as a skills document (same format as jobs)
    resume_doc = ",".join(resume_skills)
    resume_vec = vectorizer.transform([resume_doc])

    # Cosine similarity between resume and every job role
    scores = cosine_similarity(resume_vec, job_matrix).flatten()

    # Attach gap analysis for each role
    df_copy = df.copy()
    df_copy["similarity_score"] = scores
    df_copy = df_copy.sort_values("similarity_score", ascending=False).head(top_n)

    results = []
    for _, row in df_copy.iterrows():
        gap = analyze_skill_gap(resume_skills, row["skills_list"])
        results.append({
            "job_role": row["job_role"],
            "similarity_score": round(float(row["similarity_score"]), 4),
            "match_score_pct": gap["match_score"],
            "matched_skills": gap["matched_skills"],
            "missing_skills": gap["missing_skills"],
            "skill_category": row.get("skill_category", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Jaccard similarity (fast, interpretable alternative)
# ---------------------------------------------------------------------------

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard index = |A ∩ B| / |A ∪ B|"""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def recommend_jobs_jaccard(
    resume_skills: list[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Recommend jobs using Jaccard set similarity. Faster, no model needed.
    """
    df = load_dataset()
    resume_set = set(s.lower().strip() for s in resume_skills)

    results = []
    for _, row in df.iterrows():
        job_set = set(row["skills_list"])
        score = jaccard_similarity(resume_set, job_set)
        gap = analyze_skill_gap(resume_skills, row["skills_list"])
        results.append({
            "job_role": row["job_role"],
            "jaccard_score": round(score, 4),
            "match_score_pct": gap["match_score"],
            "matched_skills": gap["matched_skills"],
            "missing_skills": gap["missing_skills"],
        })

    results.sort(key=lambda x: x["jaccard_score"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Full resume analysis pipeline
# ---------------------------------------------------------------------------

def analyze_resume(resume_text: str, top_n: int = 5) -> dict:
    """
    End-to-end analysis: extract skills → recommend roles → compute gaps.

    Args:
        resume_text: Raw resume text.
        top_n: Number of job recommendations.

    Returns:
        Full analysis dict ready for API response / JSON export.
    """
    extracted_skills = extract_skills(resume_text)
    recommendations = recommend_jobs_tfidf(extracted_skills, top_n=top_n)

    return {
        "extracted_skills": extracted_skills,
        "total_skills_found": len(extracted_skills),
        "top_recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    resume = """
    Data Scientist with 3 years experience.
    Skills: Python, Pandas, NumPy, Scikit-learn, TensorFlow, SQL,
    Matplotlib, Git, Docker, Statistics, NLP, Jupyter.
    """

    result = analyze_resume(resume, top_n=5)

    print(f"\n✅ Extracted {result['total_skills_found']} skills:")
    print("   ", result["extracted_skills"])

    print("\n🎯 Top Job Recommendations:")
    for i, job in enumerate(result["top_recommendations"], 1):
        print(f"\n  {i}. {job['job_role']}")
        print(f"     Similarity: {job['similarity_score']:.2%}  |  Match: {job['match_score_pct']}%")
        print(f"     ✔ Have: {job['matched_skills']}")
        print(f"     ✖ Missing: {job['missing_skills']}")
