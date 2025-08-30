# 🚀 Deployment Guide - Resume Parser

## Quick Start for Google Colab

### Step 1: Setup Environment
```python
# Run this cell first to install all dependencies
exec(open('colab_setup.py').read())
```

### Step 2: Upload Files
Upload these files to your Colab environment:
- `app.py`
- `resume_parser.py` 
- `database_manager.py`
- `ml_models.py`
- `colab_setup.py`

### Step 3: Run Application
```python
# Start the Streamlit application
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Step 4: Access Application
Click on the external URL provided by Colab (usually ends with `.ngrok.io` or similar)

## Deployment to Streamlit Cloud

### Method 1: GitHub Repository
1. Create a new GitHub repository
2. Upload all project files
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub account
5. Select your repository
6. Set main file path: `app.py`
7. Deploy!

### Method 2: Direct Upload
1. Go to Streamlit Cloud
2. Create new app
3. Upload files directly
4. Set entry point as `app.py`
5. Deploy

## Local Development

### Setup
```bash
# Clone or download the project
cd resume-parser

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Run application
streamlit run app.py
```

## File Structure for Deployment
```
resume-parser/
├── app.py                    # Main Streamlit app
├── resume_parser.py          # Core parsing logic
├── database_manager.py       # Database operations
├── ml_models.py             # Machine learning models
├── requirements.txt         # Dependencies
├── colab_setup.py          # Colab setup script
├── setup.py                # Local setup script
├── README.md               # Documentation
├── DEPLOYMENT_GUIDE.md     # This file
└── .streamlit/
    └── config.toml         # Streamlit configuration
```

## Environment Variables (Optional)
No environment variables required - the application is self-contained.

## Troubleshooting

### Common Issues
1. **SpaCy model not found**: Run `python -m spacy download en_core_web_sm`
2. **NLTK data missing**: The app automatically downloads required data
3. **Port issues in Colab**: Use the provided command with correct port settings
4. **Memory issues**: Reduce batch sizes in ML training

### Colab Specific
- Always use `!` prefix for shell commands
- Use `--server.address 0.0.0.0` for external access
- Upload files using Colab's file upload interface

### Streamlit Cloud Specific  
- Ensure `requirements.txt` is in root directory
- Main file should be `app.py`
- No secrets or API keys needed
- SQLite database is created automatically

## Performance Optimization

### For Large Datasets
- Implement batch processing in `ml_models.py`
- Use database pagination in `database_manager.py`
- Add caching with `@st.cache_data` decorators

### Memory Management
- Clear temporary files after processing
- Use generators for large file processing
- Implement lazy loading for ML models

## Security Considerations
- File uploads are processed locally
- No external API calls required
- Database is local SQLite
- No sensitive data stored permanently
