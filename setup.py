"""
Setup script for Resume Parser project
Handles installation and setup for both local and Colab environments
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages"""
    requirements = [
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
    
    for requirement in requirements:
        print(f"Installing {requirement}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])

def download_spacy_model():
    """Download SpaCy English model"""
    print("Downloading SpaCy English model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def setup_nltk_data():
    """Download required NLTK data"""
    import nltk
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'uploads', 'temp']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main setup function"""
    print("Setting up Resume Parser project...")
    
    # Install requirements
    install_requirements()
    
    # Download SpaCy model
    download_spacy_model()
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Create directories
    create_directories()
    
    print("\n✅ Setup completed successfully!")
    print("\nTo run the application:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
