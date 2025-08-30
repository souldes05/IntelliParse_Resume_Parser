"""
Database Manager Module - Phase 3 Implementation
Satisfies Phase 3 requirements:
1. Design structured database schema to store parsed resume data
2. Populate database with extracted information
3. SQL queries and basic data analysis techniques
"""

import sqlite3
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path: str = "resume_database.db"):
        """
        Initialize database manager with structured schema
        SATISFIES REQUIREMENT: "Learn how to design a structured database schema 
        to store parsed resume data"
        """
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """
        Create database tables with proper schema design
        Tables: resumes, entities, relationships, skills, work_experience, education
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main resumes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                phone TEXT,
                original_text TEXT,
                cleaned_text TEXT,
                skills TEXT,  -- JSON string of skills array
                experience_years INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_name TEXT,
                parsing_metrics TEXT  -- JSON string of metrics
            )
        ''')
        
        # Entities table for NER results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER,
                entity_text TEXT,
                entity_label TEXT,
                start_pos INTEGER,
                end_pos INTEGER,
                confidence REAL,
                FOREIGN KEY (resume_id) REFERENCES resumes (id)
            )
        ''')
        
        # Relationships table for extracted relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER,
                subject TEXT,
                relation TEXT,
                object TEXT,
                confidence REAL,
                dependency_pattern TEXT,
                FOREIGN KEY (resume_id) REFERENCES resumes (id)
            )
        ''')
        
        # Skills table (normalized)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER,
                skill_name TEXT,
                skill_category TEXT,
                proficiency_level TEXT,
                FOREIGN KEY (resume_id) REFERENCES resumes (id)
            )
        ''')
        
        # Work experience table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS work_experience (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER,
                company_name TEXT,
                job_title TEXT,
                start_date TEXT,
                end_date TEXT,
                description TEXT,
                FOREIGN KEY (resume_id) REFERENCES resumes (id)
            )
        ''')
        
        # Education table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS education (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER,
                institution TEXT,
                degree TEXT,
                field_of_study TEXT,
                graduation_year INTEGER,
                gpa REAL,
                FOREIGN KEY (resume_id) REFERENCES resumes (id)
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_resume_email ON resumes(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_resume ON entities(resume_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_resume ON relationships(resume_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_skills_resume ON skills(resume_id)')
        
        conn.commit()
        conn.close()

    def insert_resume(self, resume_data: Dict[str, Any]) -> int:
        """
        Insert parsed resume data into database
        SATISFIES REQUIREMENT: "Practice populating the database with the extracted information"
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert main resume record
            cursor.execute('''
                INSERT OR REPLACE INTO resumes 
                (name, email, phone, original_text, cleaned_text, skills, experience_years, file_name, parsing_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                resume_data.get('name'),
                resume_data.get('email'),
                resume_data.get('phone'),
                resume_data.get('original_text', ''),
                resume_data.get('cleaned_text', ''),
                json.dumps(resume_data.get('skills', [])),
                resume_data.get('experience_years'),
                resume_data.get('file_name', 'unknown'),
                json.dumps(resume_data.get('ner_metrics', {}))
            ))
            
            resume_id = cursor.lastrowid
            
            # Insert entities
            if 'entities' in resume_data:
                for entity in resume_data['entities']:
                    cursor.execute('''
                        INSERT INTO entities 
                        (resume_id, entity_text, entity_label, start_pos, end_pos, confidence)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        resume_id,
                        entity['text'],
                        entity['label'],
                        entity.get('start', 0),
                        entity.get('end', 0),
                        entity.get('confidence', 1.0)
                    ))
            
            # Insert relationships
            if 'relationships' in resume_data:
                for rel in resume_data['relationships']:
                    cursor.execute('''
                        INSERT INTO relationships 
                        (resume_id, subject, relation, object, confidence, dependency_pattern)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        resume_id,
                        rel['subject'],
                        rel['relation'],
                        rel['object'],
                        rel.get('confidence', 1.0),
                        rel.get('dependency_pattern', '')
                    ))
            
            # Insert skills (normalized)
            if 'skills' in resume_data:
                for skill in resume_data['skills']:
                    skill_category = self._categorize_skill(skill)
                    cursor.execute('''
                        INSERT INTO skills 
                        (resume_id, skill_name, skill_category, proficiency_level)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        resume_id,
                        skill,
                        skill_category,
                        'intermediate'  # Default proficiency
                    ))
            
            conn.commit()
            return resume_id
            
        except Exception as e:
            conn.rollback()
            print(f"Error inserting resume: {str(e)}")
            return -1
        finally:
            conn.close()

    def _categorize_skill(self, skill: str) -> str:
        """Categorize skills into different types"""
        skill_lower = skill.lower()
        
        programming_languages = ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby']
        databases = ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle']
        frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express']
        cloud_platforms = ['aws', 'azure', 'gcp', 'docker', 'kubernetes']
        data_science = ['machine learning', 'data science', 'pandas', 'numpy', 'tensorflow', 'pytorch']
        
        if any(lang in skill_lower for lang in programming_languages):
            return 'Programming Language'
        elif any(db in skill_lower for db in databases):
            return 'Database'
        elif any(fw in skill_lower for fw in frameworks):
            return 'Framework'
        elif any(cloud in skill_lower for cloud in cloud_platforms):
            return 'Cloud/DevOps'
        elif any(ds in skill_lower for ds in data_science):
            return 'Data Science/ML'
        else:
            return 'Other'

    def get_all_resumes(self) -> List[Dict[str, Any]]:
        """
        Retrieve all resumes from database
        SATISFIES REQUIREMENT: "Familiarize yourself with SQL queries and basic data analysis techniques"
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, email, phone, skills, experience_years, created_at, file_name
            FROM resumes
            ORDER BY created_at DESC
        ''')
        
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            resume_dict = dict(zip(columns, row))
            # Parse JSON fields
            if resume_dict['skills']:
                resume_dict['skills'] = json.loads(resume_dict['skills'])
            results.append(resume_dict)
        
        conn.close()
        return results

    def get_resume_by_id(self, resume_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed resume information by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get main resume data
        cursor.execute('SELECT * FROM resumes WHERE id = ?', (resume_id,))
        resume_row = cursor.fetchone()
        
        if not resume_row:
            conn.close()
            return None
        
        columns = [description[0] for description in cursor.description]
        resume = dict(zip(columns, resume_row))
        
        # Parse JSON fields
        if resume['skills']:
            resume['skills'] = json.loads(resume['skills'])
        if resume['parsing_metrics']:
            resume['parsing_metrics'] = json.loads(resume['parsing_metrics'])
        
        # Get entities
        cursor.execute('SELECT * FROM entities WHERE resume_id = ?', (resume_id,))
        entities = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
        resume['entities'] = entities
        
        # Get relationships
        cursor.execute('SELECT * FROM relationships WHERE resume_id = ?', (resume_id,))
        relationships = [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
        resume['relationships'] = relationships
        
        conn.close()
        return resume

    def execute_custom_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute custom SQL queries for data analysis
        SATISFIES REQUIREMENT: "Familiarize yourself with SQL queries and basic data analysis techniques"
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return results
        except Exception as e:
            print(f"Query execution error: {str(e)}")
            return []
        finally:
            conn.close()

    def get_analytics_data(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics data for dashboard
        Demonstrates advanced SQL queries and data analysis
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        analytics = {}
        
        # Basic statistics
        cursor.execute('SELECT COUNT(*) as total_resumes FROM resumes')
        analytics['total_resumes'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(experience_years) as avg_experience FROM resumes WHERE experience_years IS NOT NULL')
        result = cursor.fetchone()[0]
        analytics['avg_experience'] = round(result, 2) if result else 0
        
        # Skills analysis
        cursor.execute('''
            SELECT skill_name, COUNT(*) as frequency 
            FROM skills 
            GROUP BY skill_name 
            ORDER BY frequency DESC 
            LIMIT 10
        ''')
        analytics['top_skills'] = [{'skill': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Experience distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN experience_years < 2 THEN 'Junior (0-2 years)'
                    WHEN experience_years < 5 THEN 'Mid-level (2-5 years)'
                    WHEN experience_years < 10 THEN 'Senior (5-10 years)'
                    ELSE 'Expert (10+ years)'
                END as experience_level,
                COUNT(*) as count
            FROM resumes 
            WHERE experience_years IS NOT NULL
            GROUP BY experience_level
        ''')
        analytics['experience_distribution'] = [{'level': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Skills by category
        cursor.execute('''
            SELECT skill_category, COUNT(*) as count 
            FROM skills 
            GROUP BY skill_category 
            ORDER BY count DESC
        ''')
        analytics['skills_by_category'] = [{'category': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Entity analysis
        cursor.execute('''
            SELECT entity_label, COUNT(*) as count 
            FROM entities 
            GROUP BY entity_label 
            ORDER BY count DESC
        ''')
        analytics['entity_distribution'] = [{'label': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Relationship analysis
        cursor.execute('''
            SELECT relation, COUNT(*) as count 
            FROM relationships 
            GROUP BY relation 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        analytics['top_relationships'] = [{'relation': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        return analytics

    def search_resumes(self, search_term: str, search_type: str = 'all') -> List[Dict[str, Any]]:
        """
        Search resumes by various criteria
        Demonstrates complex SQL queries with text search
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if search_type == 'skills':
            cursor.execute('''
                SELECT DISTINCT r.* FROM resumes r
                JOIN skills s ON r.id = s.resume_id
                WHERE s.skill_name LIKE ?
            ''', (f'%{search_term}%',))
        
        elif search_type == 'name':
            cursor.execute('SELECT * FROM resumes WHERE name LIKE ?', (f'%{search_term}%',))
        
        elif search_type == 'email':
            cursor.execute('SELECT * FROM resumes WHERE email LIKE ?', (f'%{search_term}%',))
        
        else:  # search all fields
            cursor.execute('''
                SELECT * FROM resumes 
                WHERE name LIKE ? OR email LIKE ? OR cleaned_text LIKE ?
            ''', (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Parse JSON fields
        for result in results:
            if result['skills']:
                result['skills'] = json.loads(result['skills'])
        
        conn.close()
        return results

    def get_database_schema(self) -> Dict[str, List[str]]:
        """Get database schema information for documentation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema = {}
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]
            schema[table] = columns
        
        conn.close()
        return schema

    def backup_database(self, backup_path: str = None) -> str:
        """Create a backup of the database"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"resume_database_backup_{timestamp}.db"
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        return backup_path

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metrics = {}
        
        # Database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        metrics['database_size_bytes'] = cursor.fetchone()[0]
        
        # Record counts
        for table in ['resumes', 'entities', 'relationships', 'skills']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            metrics[f'{table}_count'] = cursor.fetchone()[0]
        
        conn.close()
        return metrics

# Example usage and testing
if __name__ == "__main__":
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Test data
    sample_resume_data = {
        'name': 'John Doe',
        'email': 'john.doe@email.com',
        'phone': '(555) 123-4567',
        'original_text': 'Sample original text...',
        'cleaned_text': 'Sample cleaned text...',
        'skills': ['Python', 'Machine Learning', 'SQL'],
        'experience_years': 5,
        'file_name': 'john_doe_resume.pdf',
        'entities': [
            {'text': 'John Doe', 'label': 'PERSON', 'start': 0, 'end': 8, 'confidence': 0.99},
            {'text': 'Python', 'label': 'SKILL', 'start': 50, 'end': 56, 'confidence': 0.95}
        ],
        'relationships': [
            {'subject': 'John Doe', 'relation': 'has_skill', 'object': 'Python', 'confidence': 0.9}
        ],
        'ner_metrics': {'precision': 0.85, 'recall': 0.82, 'f1': 0.835}
    }
    
    # Insert test data
    resume_id = db_manager.insert_resume(sample_resume_data)
    print(f"Inserted resume with ID: {resume_id}")
    
    # Test queries
    all_resumes = db_manager.get_all_resumes()
    print(f"Total resumes in database: {len(all_resumes)}")
    
    # Test analytics
    analytics = db_manager.get_analytics_data()
    print(f"Analytics data: {analytics}")
    
    # Test schema
    schema = db_manager.get_database_schema()
    print(f"Database schema: {schema}")
