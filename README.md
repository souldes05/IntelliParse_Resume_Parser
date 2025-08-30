# Resume Parser & Analyzer - Phase 3 Implementation

## 📋 Project Overview

This is a comprehensive resume parsing and analysis system that satisfies all Phase 3 requirements. The system uses advanced NLP techniques, machine learning models, and provides a web interface for easy interaction.

## ✅ Phase 3 Requirements Satisfied

### Text Processing & Cleaning
- ✅ **Noise Removal**: Uses regex patterns to identify and remove irrelevant information
- ✅ **Formatting Consistency**: Handles date formats, phone numbers, emails using dateutil and regex
- ✅ **Normalization**: Implements tokenization, stemming, and lemmatization using NLTK

### Named Entity Recognition (NER)
- ✅ **SpaCy Integration**: Uses SpaCy for NER with custom model training capabilities
- ✅ **Custom NER Training**: Trains models on labeled resume data
- ✅ **Performance Evaluation**: Calculates precision, recall, and F-score metrics

### Relationship Extraction
- ✅ **Dependency Parsing**: Extracts relationships between entities using SpaCy's dependency parser
- ✅ **Multiple Techniques**: Implements various relationship extraction methods
- ✅ **Ground Truth Validation**: Validates extracted relationships against known patterns

### Database & Analytics
- ✅ **Structured Schema**: SQLite database with normalized tables for resumes, entities, relationships
- ✅ **Data Population**: Automated insertion of parsed resume data
- ✅ **SQL Queries**: Advanced querying capabilities and analytics functions

### Machine Learning
- ✅ **ML Models**: Salary prediction, job classification, and clustering models
- ✅ **Insights Generation**: Extracts patterns and makes predictions from resume data
- ✅ **Performance Metrics**: Comprehensive model evaluation and validation

## 🚀 Quick Start

### For Google Colab

1. **Install Dependencies**:
```python
!pip install streamlit spacy pandas numpy scikit-learn nltk python-dateutil regex PyPDF2 python-docx seaborn matplotlib plotly wordcloud textblob
!python -m spacy download en_core_web_sm
```

2. **Clone/Download Files**:
```python
# Upload all Python files to Colab or clone from repository
```

3. **Run the Application**:
```python
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### For Local Development

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Run Application**:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
resume-parser/
├── app.py                 # Main Streamlit application
├── resume_parser.py       # Core parsing logic with NLP
├── database_manager.py    # Database operations and schema
├── ml_models.py          # Machine learning models
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── models/              # Saved ML models (created automatically)
```

## 🔧 Core Components

### 1. ResumeParser (`resume_parser.py`)
**Satisfies Phase 3 Requirements:**
- **Text Cleaning**: `clean_and_remove_noise()` - Removes noise using regex
- **Formatting**: `handle_formatting_inconsistencies()` - Normalizes dates, phones, emails
- **Normalization**: `normalize_text()` - Tokenization, stemming, lemmatization
- **NER**: `extract_entities()` - SpaCy-based entity extraction
- **Custom NER**: `train_custom_ner_model()` - Trains custom models
- **Evaluation**: `evaluate_ner_model()` - Precision, recall, F-score
- **Relationships**: `extract_relationships()` - Dependency parsing for relationships
- **Validation**: `validate_relationships()` - Ground truth validation

### 2. DatabaseManager (`database_manager.py`)
**Satisfies Phase 3 Requirements:**
- **Schema Design**: Structured tables for resumes, entities, relationships
- **Data Population**: `insert_resume()` - Populates database with parsed data
- **SQL Queries**: `execute_custom_query()` - Advanced SQL operations
- **Analytics**: `get_analytics_data()` - Data analysis functions

### 3. MLModels (`ml_models.py`)
**Satisfies Phase 3 Requirements:**
- **ML Training**: `train_models()` - Trains multiple ML models
- **Salary Prediction**: Random Forest regression model
- **Job Classification**: Text classification using TF-IDF and Random Forest
- **Clustering**: K-means clustering for resume segmentation
- **Insights**: `generate_insights()` - Extracts patterns and predictions

### 4. Streamlit App (`app.py`)
**Features:**
- File upload (PDF, DOCX, TXT)
- Real-time parsing and analysis
- Interactive database viewer
- Analytics dashboard with visualizations
- ML predictions interface
- Model performance metrics

## 📊 Features Breakdown

### Text Processing Pipeline
1. **File Extraction** → PDF/DOCX/TXT to text
2. **Noise Removal** → Regex-based cleaning
3. **Format Normalization** → Consistent date/phone/email formats
4. **Text Normalization** → Tokenization, stemming, lemmatization

### NLP & NER Pipeline
1. **Entity Extraction** → Person, organization, skill, experience entities
2. **Custom Training** → Trains NER models on labeled data
3. **Performance Evaluation** → Precision, recall, F-score metrics
4. **Relationship Extraction** → Subject-verb-object relationships using dependency parsing

### Database Operations
1. **Structured Storage** → Normalized database schema
2. **Data Population** → Automated insertion of parsed data
3. **Query Interface** → Custom SQL query execution
4. **Analytics Functions** → Built-in data analysis capabilities

### Machine Learning
1. **Salary Prediction** → Predicts salary based on skills and experience
2. **Job Classification** → Classifies job roles from resume text
3. **Resume Clustering** → Groups similar resumes together
4. **Performance Metrics** → Model evaluation and validation

## 🎯 Usage Examples

### Upload and Parse Resume
1. Navigate to "Upload & Parse" page
2. Upload PDF/DOCX/TXT resume files
3. View parsing results including:
   - Cleaned text comparison
   - Extracted entities (NER results)
   - Relationship extraction results
   - Structured information (name, email, skills, etc.)

### Database Analysis
1. Navigate to "Database View" page
2. View all parsed resumes
3. Execute custom SQL queries
4. Analyze resume statistics

### ML Predictions
1. Navigate to "ML Predictions" page
2. Get salary predictions based on experience and skills
3. Classify job roles from resume text
4. View clustering analysis results

### Performance Monitoring
1. Navigate to "Model Performance" page
2. View NER model metrics (precision, recall, F-score)
3. Test NER model with custom text
4. Analyze relationship extraction performance

## 🚀 Deployment to Streamlit Cloud

### Via GitHub
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from repository
4. No additional configuration needed

### Requirements for Deployment
- All dependencies listed in `requirements.txt`
- No external API keys required
- Uses SQLite (no external database needed)
- Self-contained ML models

## 📈 Performance Metrics

The system provides comprehensive metrics for all components:

- **NER Model**: Precision, Recall, F1-Score
- **Relationship Extraction**: Accuracy, validation against ground truth
- **ML Models**: RMSE for regression, accuracy for classification
- **Database**: Query performance and storage metrics

## 🔍 Code Quality Features

- **Comprehensive Documentation**: Every function documented with purpose
- **Error Handling**: Robust error handling throughout
- **Modular Design**: Separate modules for different functionalities
- **Type Hints**: Python type hints for better code clarity
- **Performance Optimization**: Efficient algorithms and caching

## 🛠️ Troubleshooting

### Common Issues
1. **SpaCy Model Missing**: Run `python -m spacy download en_core_web_sm`
2. **NLTK Data Missing**: The code automatically downloads required NLTK data
3. **Memory Issues**: Use smaller batch sizes for large datasets
4. **File Upload Issues**: Ensure files are in supported formats (PDF, DOCX, TXT)

### For Colab Specific Issues
- Use `!streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
- Install dependencies with `!pip install` prefix
- Upload files directly to Colab environment

## 📝 Phase 3 Requirements Mapping

| Requirement | Implementation | File | Function |
|-------------|----------------|------|----------|
| Noise removal with regex | ✅ | `resume_parser.py` | `clean_and_remove_noise()` |
| Format inconsistencies handling | ✅ | `resume_parser.py` | `handle_formatting_inconsistencies()` |
| Normalization techniques | ✅ | `resume_parser.py` | `normalize_text()` |
| NER with SpaCy | ✅ | `resume_parser.py` | `extract_entities()` |
| Custom NER training | ✅ | `resume_parser.py` | `train_custom_ner_model()` |
| NER evaluation metrics | ✅ | `resume_parser.py` | `evaluate_ner_model()` |
| Relationship extraction | ✅ | `resume_parser.py` | `extract_relationships()` |
| Dependency parsing | ✅ | `resume_parser.py` | `extract_relationships()` |
| Relationship validation | ✅ | `resume_parser.py` | `validate_relationships()` |
| Database schema design | ✅ | `database_manager.py` | `init_database()` |
| Database population | ✅ | `database_manager.py` | `insert_resume()` |
| SQL queries & analysis | ✅ | `database_manager.py` | `execute_custom_query()` |
| ML models for insights | ✅ | `ml_models.py` | `train_models()` |

## 🎉 Success Criteria Met

✅ **All Phase 3 objectives completed**  
✅ **Runs without errors on Colab**  
✅ **Deployable to Streamlit Cloud**  
✅ **No ngrok dependency**  
✅ **Comprehensive documentation**  
✅ **Step-by-step code explanation**  
✅ **Production-ready code quality**
