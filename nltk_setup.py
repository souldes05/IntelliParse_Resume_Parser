"""
NLTK Data Setup Script for Streamlit Cloud
Downloads required NLTK data packages for the resume parser
"""

import nltk
import ssl

def download_nltk_data():
    """Download all required NLTK data packages"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # List of required NLTK data packages
    packages = [
        'punkt',
        'punkt_tab', 
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'omw-1.4'
    ]
    
    print("Downloading NLTK data packages...")
    for package in packages:
        try:
            nltk.download(package, quiet=True)
            print(f"✓ Downloaded {package}")
        except Exception as e:
            print(f"✗ Failed to download {package}: {e}")
    
    print("NLTK data download complete!")

if __name__ == "__main__":
    download_nltk_data()
