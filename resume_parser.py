"""
Resume Parser Module - Deployment-Ready Implementation
Uses NLTK with robust fallback mechanisms for Streamlit Cloud deployment.
Implements all core NLP functionality without SpaCy dependencies.
"""

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from dateutil import parser as date_parser
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import PyPDF2
import docx
from textblob import TextBlob
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

class ResumeParser:
    """
    Resume Parser with robust NLTK-based NLP processing
    Designed for deployment on Streamlit Cloud without compilation dependencies
    """
    
    def __init__(self):
        """Initialize the resume parser with all required components"""
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize stopwords with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback to basic stopwords if download fails
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did'}
        
        # Define patterns for information extraction
        self._initialize_patterns()
        
        # Skills database
        self.skills_database = self._load_skills_database()
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        required_data = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
        for data in required_data:
            try:
                if data == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif data == 'punkt_tab':
                    nltk.data.find('tokenizers/punkt_tab')
                elif data == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif data == 'wordnet':
                    nltk.data.find('corpora/wordnet')
                else:
                    nltk.data.find(f'taggers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except Exception as e:
                    print(f"Warning: Could not download {data}: {e}")
                    continue
    
    def _initialize_patterns(self):
        """Initialize regex patterns for information extraction"""
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'linkedin': r'linkedin\.com/in/[\w-]+',
            'github': r'github\.com/[\w-]+',
            'degree': r'\b(Bachelor|Master|PhD|B\.S\.|M\.S\.|B\.A\.|M\.A\.|MBA|Ph\.D\.)\b',
            'years_exp': r'(\d+)\+?\s*years?\s*(of\s*)?experience',
            'gpa': r'GPA[:\s]*(\d+\.?\d*)',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\w+\s+\d{1,2},?\s+\d{4}\b'
        }
    
    def _load_skills_database(self):
        """Load comprehensive skills database"""
        return {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
                'go', 'rust', 'scala', 'r', 'matlab', 'perl', 'typescript', 'dart', 'julia'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django',
                'flask', 'spring', 'laravel', 'bootstrap', 'jquery', 'sass', 'webpack'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'sqlite', 'redis', 'cassandra',
                'oracle', 'sql server', 'dynamodb', 'elasticsearch'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'terraform',
                'jenkins', 'gitlab', 'github actions'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter', 'tableau', 'power bi'
            ]
        }

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        text = ""
        file_extension = file_path.lower().split('.')[-1]
        
        try:
            if file_extension == 'pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            
            elif file_extension == 'docx':
                doc = docx.Document(file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + '\n'
            
            elif file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return ""
        
        return text

    def preprocess_text(self, text):
        """
        Advanced text preprocessing pipeline with fallback mechanisms
        """
        # Basic cleaning
        text = re.sub(r'[^\w\s@.-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization with fallback
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            # Fallback to simple split tokenization
            tokens = text.lower().split()
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatization with fallback
        try:
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        except LookupError:
            # Fallback to stemming if lemmatization fails
            lemmatized_tokens = [self.stemmer.stem(token) for token in tokens]
        
        return {
            'original_text': text,
            'tokens': tokens,
            'lemmatized_tokens': lemmatized_tokens,
            'processed_text': ' '.join(lemmatized_tokens)
        }
    
    def extract_named_entities(self, text):
        """
        Extract named entities using NLTK with fallback mechanisms
        """
        entities = {
            'PERSON': [],
            'ORGANIZATION': [],
            'GPE': [],  # Geopolitical entities
            'DATE': [],
            'MONEY': []
        }
        
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity chunking
            tree = ne_chunk(pos_tags)
            
            for chunk in tree:
                if hasattr(chunk, 'label'):
                    entity_name = ' '.join([token for token, pos in chunk.leaves()])
                    entity_type = chunk.label()
                    if entity_type in entities:
                        entities[entity_type].append(entity_name)
        except LookupError:
            # Fallback to regex-based entity extraction
            # Extract potential names (capitalized words)
            name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
            entities['PERSON'] = re.findall(name_pattern, text)
            
            # Extract potential organizations (words with Inc, Corp, LLC, etc.)
            org_pattern = r'\b[A-Z][a-zA-Z\s]*(?:Inc|Corp|LLC|Company|Ltd|Corporation)\b'
            entities['ORGANIZATION'] = re.findall(org_pattern, text)
        
        return entities

    def extract_contact_info(self, text):
        """Extract contact information using regex patterns"""
        contact_info = {}
        
        # Email
        email_matches = re.findall(self.patterns['email'], text, re.IGNORECASE)
        contact_info['email'] = email_matches[0] if email_matches else None
        
        # Phone
        phone_matches = re.findall(self.patterns['phone'], text)
        contact_info['phone'] = phone_matches[0] if phone_matches else None
        
        # LinkedIn
        linkedin_matches = re.findall(self.patterns['linkedin'], text, re.IGNORECASE)
        contact_info['linkedin'] = linkedin_matches[0] if linkedin_matches else None
        
        # GitHub
        github_matches = re.findall(self.patterns['github'], text, re.IGNORECASE)
        contact_info['github'] = github_matches[0] if github_matches else None
        
        return contact_info

    def extract_education(self, text):
        """Extract education information with fallback sentence tokenization"""
        education = []
        
        # Find degree mentions
        degree_matches = re.findall(self.patterns['degree'], text, re.IGNORECASE)
        
        # Find GPA
        gpa_matches = re.findall(self.patterns['gpa'], text, re.IGNORECASE)
        
        # Extract university names (simplified approach)
        university_keywords = ['university', 'college', 'institute', 'school']
        
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback to simple sentence splitting
            sentences = text.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in university_keywords):
                if any(degree in sentence for degree in degree_matches):
                    education.append({
                        'degree': degree_matches[0] if degree_matches else 'Unknown',
                        'institution': sentence.strip(),
                        'gpa': gpa_matches[0] if gpa_matches else None
                    })
                    break
        
        return education

    def extract_skills(self, text):
        """Extract skills from text using comprehensive database"""
        found_skills = []
        text_lower = text.lower()
        
        for category, skills in self.skills_database.items():
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills.append(skill)
        
        return list(set(found_skills))  # Remove duplicates

    def extract_experience(self, text):
        """Extract work experience information with fallback tokenization"""
        experience = []
        
        # Find years of experience
        years_matches = re.findall(self.patterns['years_exp'], text, re.IGNORECASE)
        total_years = int(years_matches[0]) if years_matches else 0
        
        # Extract job titles and companies (simplified)
        job_keywords = ['engineer', 'developer', 'analyst', 'manager', 'director', 'specialist']
        
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback to simple sentence splitting
            sentences = text.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in job_keywords):
                experience.append({
                    'title': sentence.strip()[:100],  # Limit length
                    'years': total_years
                })
                break
        
        return experience, total_years

    def extract_relationships(self, text):
        """
        Extract relationships between entities with fallback tokenization
        """
        relationships = []
        
        # Simple relationship extraction based on proximity
        entities = self.extract_named_entities(text)
        
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback to simple sentence splitting
            sentences = text.split('.')
        
        for sentence in sentences:
            sentence_entities = []
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity.lower() in sentence.lower():
                        sentence_entities.append((entity, entity_type))
            
            # If multiple entities in same sentence, create relationship
            if len(sentence_entities) > 1:
                for i in range(len(sentence_entities)):
                    for j in range(i+1, len(sentence_entities)):
                        relationships.append({
                            'entity1': sentence_entities[i][0],
                            'entity1_type': sentence_entities[i][1],
                            'entity2': sentence_entities[j][0],
                            'entity2_type': sentence_entities[j][1],
                            'context': sentence.strip()
                        })
        
        return relationships

    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            return {
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity,
                'label': 'positive' if sentiment.polarity > 0 else 'negative' if sentiment.polarity < 0 else 'neutral'
            }
        except Exception as e:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'label': 'neutral'
            }

    def parse_resume(self, file_path):
        """Main parsing function that processes a resume file"""
        raw_text = self.extract_text_from_file(file_path)
        if not raw_text:
            return None
        
        processed_data = self.preprocess_text(raw_text)
        contact_info = self.extract_contact_info(raw_text)
        education = self.extract_education(raw_text)
        skills = self.extract_skills(raw_text)
        experience, years_exp = self.extract_experience(raw_text)
        entities = self.extract_named_entities(raw_text)
        relationships = self.extract_relationships(raw_text)
        sentiment = self.analyze_sentiment(raw_text)
        
        parsed_data = {
            'file_path': file_path,
            'raw_text': raw_text,
            'processed_text': processed_data['processed_text'],
            'tokens': processed_data['tokens'],
            'lemmatized_tokens': processed_data['lemmatized_tokens'],
            'contact_info': contact_info,
            'education': education,
            'skills': skills,
            'experience': experience,
            'years_experience': years_exp,
            'named_entities': entities,
            'relationships': relationships,
            'sentiment': sentiment,
            'parsing_timestamp': datetime.now().isoformat(),
            'word_count': len(processed_data['tokens']),
            'skill_count': len(skills),
            'education_count': len(education),
            'experience_count': len(experience)
        }
        
        return parsed_data
