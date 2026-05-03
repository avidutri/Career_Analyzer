"""
utils/skill_extraction.py
--------------------------
Extracts skills from resume text using two approaches:
  1. Keyword matching against a skill dictionary (primary)
  2. spaCy NER tagging (optional, used if spaCy is installed)

Skill normalization maps aliases to canonical names,
e.g., "ml" -> "machine learning", "js" -> "javascript".
"""

import re
from utils.text_preprocessing import clean_text

# ---------------------------------------------------------------------------
# Master skill dictionary  (add more as needed)
# ---------------------------------------------------------------------------

SKILL_DICTIONARY = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    "bash", "shell", "perl", "dart", "lua", "solidity",

    # Web / Frontend
    "html", "css", "react", "angular", "vue", "redux", "webpack", "sass",
    "tailwind", "bootstrap", "next.js", "nuxt", "svelte", "jquery",
    "responsive design", "figma", "sketch", "adobe xd", "wireframing",
    "prototyping",

    # Backend / APIs
    "django", "flask", "fastapi", "express", "spring", "rails", "laravel",
    "rest api", "graphql", "grpc", "websocket", "microservices",

    # Databases
    "sql", "postgresql", "mysql", "sqlite", "mongodb", "redis", "cassandra",
    "elasticsearch", "neo4j", "oracle", "dynamodb", "firebase",

    # ML / AI
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    "transformers", "bert", "gpt", "hugging face", "opencv", "spacy", "nltk",
    "statistics", "linear regression", "logistic regression", "random forest",
    "neural network", "cnn", "rnn", "lstm", "reinforcement learning",
    "feature engineering", "model deployment", "mlops", "experiment design",

    # Data
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau",
    "power bi", "excel", "data visualization", "data analysis",
    "data warehousing", "etl", "data modeling", "spark", "hadoop",
    "kafka", "airflow", "dbt", "business intelligence",

    # DevOps / Cloud
    "docker", "kubernetes", "ci/cd", "jenkins", "git", "github", "gitlab",
    "aws", "azure", "gcp", "terraform", "ansible", "linux", "nginx",
    "monitoring", "logging", "prometheus", "grafana", "serverless",

    # Mobile
    "android sdk", "ios sdk", "react native", "flutter", "xcode",
    "jetpack compose", "swiftui", "firebase",

    # Security
    "penetration testing", "vulnerability assessment", "cryptography",
    "firewalls", "siem", "incident response", "networking", "vpn",

    # Project / Process
    "agile", "scrum", "jira", "kanban", "product strategy", "roadmapping",
    "stakeholder management", "user research", "usability testing",
    "design thinking", "communication", "academic writing", "latex",

    # Blockchain
    "ethereum", "web3", "smart contracts", "solidity", "truffle",
    "hardhat", "defi", "nft",

    # Testing
    "selenium", "jest", "pytest", "unit testing", "api testing", "postman",
    "test automation",
}

# ---------------------------------------------------------------------------
# Alias / normalization map  →  canonical skill name
# ---------------------------------------------------------------------------

ALIAS_MAP = {
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "machine learning",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "tf": "tensorflow",
    "sk-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "sk learn": "scikit-learn",
    "hf": "hugging face",
    "k8s": "kubernetes",
    "k 8s": "kubernetes",
    "gh": "github",
    "pg": "postgresql",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "nosql": "mongodb",
    "react.js": "react",
    "reactjs": "react",
    "node": "express",
    "node.js": "express",
    "nodejs": "express",
    "next js": "next.js",
    "nextjs": "next.js",
    "vue.js": "vue",
    "vuejs": "vue",
    "angular.js": "angular",
    "angularjs": "angular",
    "cv": "computer vision",
    "ner": "nlp",
    "natural language processing": "nlp",
    "statistical analysis": "statistics",
    "data science": "machine learning",
    "rest": "rest api",
    "restful": "rest api",
    "api": "rest api",
    "devops": "ci/cd",
    "cloud": "aws",
    "version control": "git",
    "c plus plus": "c++",
    "objective-c": "ios sdk",
    "obj-c": "ios sdk",
    "android": "android sdk",
    "ios": "ios sdk",
    "power bi": "power bi",
    "powerbi": "power bi",
    "tableau": "tableau",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
}

# ---------------------------------------------------------------------------
# spaCy optional setup
# ---------------------------------------------------------------------------

try:
    import spacy
    try:
        NLP_MODEL = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def normalize_skill(skill: str) -> str:
    """Resolve aliases to canonical skill names."""
    skill = skill.lower().strip()
    return ALIAS_MAP.get(skill, skill)


def extract_skills_keyword(text: str) -> list[str]:
    """
    Primary extraction: scan cleaned text for known skill keywords.
    Handles multi-word skills (e.g., 'machine learning', 'rest api').
    """
    cleaned = clean_text(text)
    found = set()

    # Check aliases first (they may map to multi-word skills)
    for alias, canonical in ALIAS_MAP.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, cleaned):
            found.add(canonical)

    # Check full skill dictionary (longest first to catch multi-word)
    sorted_skills = sorted(SKILL_DICTIONARY, key=len, reverse=True)
    for skill in sorted_skills:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, cleaned):
            found.add(skill)

    return sorted(found)


def extract_skills_spacy(text: str) -> list[str]:
    """
    Optional: use spaCy NER to surface named entities, then cross-check
    against our dictionary. Falls back gracefully if spaCy unavailable.
    """
    if not SPACY_AVAILABLE:
        return []

    doc = NLP_MODEL(text)
    candidates = set()

    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "GPE", "WORK_OF_ART"}:
            candidates.add(ent.text.lower().strip())

    # Filter candidates that appear in skill dictionary or alias map
    found = set()
    for c in candidates:
        norm = normalize_skill(c)
        if norm in SKILL_DICTIONARY:
            found.add(norm)

    return sorted(found)


def extract_skills(text: str, use_spacy: bool = True) -> list[str]:
    """
    Unified skill extraction: keyword matching + optional spaCy NER.
    Results are merged and deduplicated.

    Args:
        text: Raw resume or description text.
        use_spacy: Whether to also run spaCy NER extraction.

    Returns:
        Sorted list of canonical skill names found in text.
    """
    keyword_skills = set(extract_skills_keyword(text))
    spacy_skills = set(extract_skills_spacy(text)) if use_spacy else set()
    all_skills = keyword_skills | spacy_skills
    return sorted(all_skills)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    resume = """
    Senior Data Scientist with 5 years of experience.
    - Proficient in Python, TensorFlow, PyTorch, and scikit-learn.
    - Built NLP pipelines using BERT and HuggingFace Transformers.
    - Strong SQL and PostgreSQL skills; experience with Spark and Airflow.
    - Deployed ML models using Docker and Kubernetes on AWS.
    - Version control via Git/GitHub. Familiar with CI/CD pipelines.
    - Statistics, feature engineering, model evaluation.
    """

    skills = extract_skills(resume)
    print(f"Extracted {len(skills)} skills:")
    for s in skills:
        print(f"  • {s}")
