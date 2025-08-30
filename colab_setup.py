"""
Google Colab Setup Script
Run this in Google Colab to set up the Resume Parser project
"""

# Install all required packages
print("📦 Installing required packages...")
import subprocess
import sys

packages = [
    "streamlit==1.28.1",
    "spacy==3.7.2", 
    "pandas==2.0.3",
    "numpy==1.24.3",
    "scikit-learn==1.3.0",
    "nltk==3.8.1",
    "python-dateutil==2.8.2",
    "regex==2023.8.8",
    "PyPDF2==3.0.1",
    "python-docx==0.8.11",
    "seaborn==0.12.2",
    "matplotlib==3.7.2",
    "plotly==5.15.0",
    "wordcloud==1.9.2",
    "textblob==0.17.1",
    "joblib==1.3.2"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Download SpaCy model
print("🔤 Downloading SpaCy English model...")
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Setup NLTK data
print("📚 Setting up NLTK data...")
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("✅ Setup completed! You can now run the Resume Parser.")
print("\nNext steps:")
print("1. Upload all Python files (app.py, resume_parser.py, database_manager.py, ml_models.py)")
print("2. Run: !streamlit run app.py --server.port 8501 --server.address 0.0.0.0")
print("3. Click on the external URL to access the application")
