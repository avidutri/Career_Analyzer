# AI-Powered Career Skill Gap Analyzer

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![License](https://img.shields.io/badge/license-MIT-blue)

An intelligent NLP + ML pipeline that analyzes resumes and provides actionable career insights.

## ✨ Key Features

- **🎯 Resume Skill Extraction** — Parse skills from PDF, DOCX, or plain text resumes using regex keyword matching + spaCy NER
- **📊 Skill Gap Detection** — Identify missing skills between your resume and target job roles
- **🔍 Job Recommendations** — Get personalized job role suggestions based on your skills using TF-IDF cosine similarity
- **⚡ REST API** — FastAPI server with interactive Swagger UI for easy integration
- **📈 Interactive Analysis** — Jupyter notebook for exploratory data analysis and model experimentation

---

## � Quick Start (5 minutes)

```bash
# 1. Clone & navigate
git clone <your-repo>
cd career_analyzer

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the API server
uvicorn api.main:app --reload --port 8000

# For production deployment (matches Render setup):
# gunicorn app:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 1 --threads 8 --timeout 0

# 5. Open browser
# Visit: http://localhost:8000/docs
```

---

## �📁 Project Structure

```
career_analyzer/
├── data/
│   └── jobs_dataset.csv         # 20 job roles with required skills
├── models/
│   └── tfidf_model.pkl          # Saved TF-IDF model (auto-generated)
├── utils/
│   ├── text_preprocessing.py    # Tokenize / stopwords / lemmatize
│   ├── skill_extraction.py      # Keyword matching + spaCy NER
│   └── similarity.py            # TF-IDF + Jaccard + gap analysis
├── api/
│   └── main.py                  # FastAPI server
├── notebooks/
│   └── analysis.ipynb           # Interactive experiments
├── outputs/
│   └── (results saved here)
└── requirements.txt
```

---

## ⚙️ Detailed Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) PDF/DOCX file for testing

### Installation Steps

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install spaCy model for advanced NER
pip install spacy
python -m spacy download en_core_web_sm
```

### Dependencies

The `requirements.txt` includes:
- **FastAPI** — Web framework
- **uvicorn** — ASGI server
- **pandas** — Data manipulation
- **scikit-learn** — TF-IDF vectorization
- **spacy** — NLP & Named Entity Recognition
- **PyPDF2** — PDF text extraction
- **python-docx** — DOCX text extraction
- **nltk** — Tokenization & lemmatization

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| spaCy model not found | Run `python -m spacy download en_core_web_sm` |
| Port 8000 already in use | Use `--port 8001` or kill the process on 8000 |
| PDF extraction fails | Ensure PyPDF2 is installed: `pip install PyPDF2` |

---

## 🚀 Run the API

### Development (with auto-reload)
```bash
# From the career_analyzer/ directory:
uvicorn api.main:app --reload --port 8000
```

### Production (matches Render deployment)
```bash
# Using gunicorn with uvicorn workers:
gunicorn app:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 1 --threads 8 --timeout 0
```

The `--reload` flag enables auto-reload during development.
Visit: **http://localhost:8000/docs** for interactive Swagger UI documentation.

---

## 📡 API Endpoints

### Health Check
Check if the API is running:
```bash
curl http://localhost:8000/health
```

### List All Job Roles
Retrieve all available job positions:
```bash
curl http://localhost:8000/job_roles
```

### Analyze Resume (Text Input)
Extract skills from plain text:
```bash
curl -X POST http://localhost:8000/analyze_resume \
  -F "resume_text=Python, TensorFlow, SQL, scikit-learn, NLP, Docker, Git" \
  -F "top_n=5"
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze_resume",
    data={
        "resume_text": "Python, Machine Learning, SQL, Docker, Git, Kubernetes",
        "top_n": 3
    }
)
print(response.json())
```

### Analyze Resume (File Upload)
Extract skills from PDF or DOCX file:
```bash
curl -X POST http://localhost:8000/upload_resume \
  -F "file=@my_resume.pdf"
```

**Python Example:**
```python
import requests

with open("my_resume.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/upload_resume",
        files=files
    )
print(response.json())
```

### Skill Gap Analysis
Find missing skills for a specific job role:
```bash
curl -X POST http://localhost:8000/skill_gap \
  -H "Content-Type: application/json" \
  -d '{
    "resume_skills": ["python", "sql", "tensorflow"],
    "job_role": "Data Scientist"
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/skill_gap",
    json={
        "resume_skills": ["python", "sql", "tensorflow"],
        "job_role": "Data Scientist"
    }
)
print(response.json())
```

### Job Recommendations
Get recommended job roles based on your skills:
```bash
curl -X POST http://localhost:8000/recommend_jobs \
  -H "Content-Type: application/json" \
  -d '{
    "skills": ["python", "machine learning", "sql", "docker"],
    "top_n": 5
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/recommend_jobs",
    json={
        "skills": ["python", "machine learning", "sql", "docker"],
        "top_n": 5
    }
)
results = response.json()
for job in results["top_recommendations"]:
    print(f"{job['job_role']}: {job['similarity_score']:.2%}")
```

---

## 📊 Response Examples

### Successful Analysis
```json
{
  "extracted_skills": ["docker", "git", "machine learning", "nlp", "python", "scikit-learn", "sql", "tensorflow"],
  "total_skills_found": 8,
  "top_recommendations": [
    {
      "job_role": "Data Scientist",
      "similarity_score": 0.7812,
      "match_score_pct": 70.0,
      "matched_skills": ["machine learning", "python", "scikit-learn", "sql", "tensorflow"],
      "missing_skills": ["data visualization", "deep learning", "numpy", "pandas", "statistics"]
    },
    {
      "job_role": "ML Engineer",
      "similarity_score": 0.6845,
      "match_score_pct": 68.45,
      "matched_skills": ["machine learning", "python", "sql", "tensorflow"],
      "missing_skills": ["kubernetes", "mlops", "pytorch", "spark"]
    }
  ]
}
```

---

## 🔬 Using Jupyter Notebook

For interactive exploration and data analysis:

```bash
pip install jupyter
cd notebooks/
jupyter notebook analysis.ipynb
```

Explore skill distributions, test different algorithms, and visualize job role similarities.

---

## 🧠 How It Works

The pipeline follows a structured NLP + ML approach:

| Step | Description | Technology |
|------|-------------|-------------|
| **1. Text Extraction** | Extract raw text from resume files (PDF/DOCX/text) | PyPDF2, python-docx |
| **2. Preprocessing** | Lowercase → tokenize → remove stopwords → lemmatize | NLTK |
| **3. Skill Recognition** | Match extracted tokens against skill dictionary + use NER | Regex + spaCy |
| **4. Vectorization** | Convert resume & job descriptions into TF-IDF vectors | scikit-learn |
| **5. Similarity Scoring** | Calculate cosine similarity between vectors | numpy |
| **6. Gap Analysis** | Find missing skills (set difference) | Python sets |
| **7. Ranking & Results** | Sort by score and return top-N recommendations | pandas |

---

## ⚙️ Configuration & Customization

### Add New Skills
Edit the skill dictionary in [utils/skill_extraction.py](utils/skill_extraction.py):
```python
SKILL_DICTIONARY = {
    "python": ["py", "python3", "python 3"],
    "javascript": ["js", "node", "nodejs"],
    # Add more skills here...
}
```

### Add New Job Roles
Update the dataset in [data/jobs_dataset.csv](data/jobs_dataset.csv):
```csv
job_role,required_skills
Data Scientist,"python, SQL, machine learning, statistics"
Backend Engineer,"python, FastAPI, Docker, PostgreSQL"
# Add more roles...
```

### Advanced Configuration
- **Skill matching algorithm**: Edit `similarity.py` to use different similarity metrics
- **Embeddings**: Replace TF-IDF with `sentence-transformers` for semantic similarity
- **NER model**: Use a different spaCy model or fine-tune for domain-specific skills

---

## 📈 Performance Notes

- **Resume processing**: ~100ms for typical resume (1-2 pages)
- **Job recommendation**: ~50ms for recommending top-10 matches
- **Scalability**: Efficiently handles 100+ job roles with batch similarity computation
- **Memory usage**: ~150MB with all models loaded

---

## 🔮 Future Enhancements

- [ ] Web UI dashboard for visualization
- [ ] Resume upload & profile storage
- [ ] Integration with LinkedIn API
- [ ] Salary prediction based on skills
- [ ] Learning path recommendations
- [ ] Real-time job market trends
- [ ] Multi-language support

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🆘 Support & Troubleshooting

### Common Issues

**Q: API returns 500 error**
- Check server logs for details
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify spaCy model is installed (if using NER)

**Q: Resume file upload fails**
- Ensure file is PDF, DOCX, or text format
- File size should be under 10MB
- Check file permissions

**Q: No skills detected from resume**
- Add more skills to `SKILL_DICTIONARY` in `skill_extraction.py`
- Enable spaCy NER for entity recognition
- Ensure resume uses standard skill names

### Getting Help

- Open an [issue](../../issues) on GitHub
- Check existing documentation in `notebooks/`
- Review FastAPI docs: http://localhost:8000/docs

---

## 🙏 Acknowledgments

Built with ❤️ using:
- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/)
- [spaCy](https://spacy.io/)
- [NLTK](https://www.nltk.org/)

---

**Last Updated**: May 2026  
**Python Version**: 3.8+  
**Status**: Active Development
# CareerAnalyzer_MiniProject
