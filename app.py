import streamlit as st
import pandas as pd
from resume_parser import ResumeParser
from database_manager import DatabaseManager
from ml_models import MLModels
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import base64

# Download NLTK data on startup
try:
    from nltk_setup import download_nltk_data
    download_nltk_data()
except Exception as e:
    st.warning(f"NLTK data download warning: {e}")

# Page configuration
st.set_page_config(
    page_title="Resume Parser & Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize parser, database, and ML models"""
    parser = ResumeParser()
    db_manager = DatabaseManager()
    ml_models = MLModels()
    return parser, db_manager, ml_models

def main():
    st.title("📄 Advanced Resume Parser & Analyzer")
    st.markdown("### Phase 3: Text Processing, NER, Relationship Extraction & ML Analysis")
    
    # Initialize components
    parser, db_manager, ml_models = initialize_components()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Upload & Parse", "Database View", "Analytics Dashboard", "ML Predictions", "Model Performance"]
    )
    
    if page == "Upload & Parse":
        upload_and_parse_page(parser, db_manager)
    elif page == "Database View":
        database_view_page(db_manager)
    elif page == "Analytics Dashboard":
        analytics_dashboard_page(db_manager)
    elif page == "ML Predictions":
        ml_predictions_page(ml_models, db_manager)
    elif page == "Model Performance":
        model_performance_page(parser)

def upload_and_parse_page(parser, db_manager):
    st.header("📤 Upload & Parse Resumes")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")
            
            # Save uploaded file temporarily
            with open(f"temp_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Parse resume
            with st.spinner("Parsing resume..."):
                result = parser.parse_resume(f"temp_{uploaded_file.name}")
            
            if result:
                # Display parsing results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🧹 Text Cleaning Results")
                    st.text_area("Original Text (first 500 chars)", 
                               result.get('raw_text', '')[:500], height=150)
                    st.text_area("Processed Text (first 500 chars)", 
                               result.get('processed_text', '')[:500], height=150)
                
                with col2:
                    st.subheader("📊 Extracted Information")
                    contact_info = result.get('contact_info', {})
                    st.json({
                        "Email": contact_info.get('email', 'Not found'),
                        "Phone": contact_info.get('phone', 'Not found'),
                        "LinkedIn": contact_info.get('linkedin', 'Not found'),
                        "Skills": result.get('skills', [])[:5],  # Show first 5 skills
                        "Experience Years": result.get('years_experience', 'Not calculated')
                    })
                
                # NER Results
                st.subheader("🏷️ Named Entity Recognition Results")
                named_entities = result.get('named_entities', {})
                if named_entities:
                    for entity_type, entities in named_entities.items():
                        if entities:
                            st.write(f"**{entity_type}:** {', '.join(entities[:3])}")
                
                # Relationships
                st.subheader("🔗 Extracted Relationships")
                relationships = result.get('relationships', [])
                if relationships:
                    for rel in relationships[:5]:  # Show first 5 relationships
                        st.write(f"**{rel.get('entity1', 'Unknown')}** → *{rel.get('entity1_type', 'related to')}* → **{rel.get('entity2', 'Unknown')}**")
                
                # Save to database
                if st.button(f"Save {uploaded_file.name} to Database"):
                    db_manager.insert_resume(result)
                    st.success("Resume saved to database!")
            
            # Clean up temp file
            import os
            if os.path.exists(f"temp_{uploaded_file.name}"):
                os.remove(f"temp_{uploaded_file.name}")

def database_view_page(db_manager):
    st.header("🗄️ Database View")
    
    # Get all resumes
    resumes = db_manager.get_all_resumes()
    
    if resumes:
        df = pd.DataFrame(resumes)
        st.subheader(f"📋 Total Resumes: {len(df)}")
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Resumes", len(df))
        with col2:
            avg_exp = df['experience_years'].mean() if 'experience_years' in df.columns else 0
            st.metric("Avg Experience", f"{avg_exp:.1f} years")
        with col3:
            unique_skills = len(set([skill for skills in df['skills'] if skills for skill in skills.split(',')]))
            st.metric("Unique Skills", unique_skills)
        
        # Display resumes table
        st.subheader("📊 Resumes Data")
        st.dataframe(df)
        
        # SQL Query Interface
        st.subheader("🔍 Custom SQL Queries")
        query = st.text_area("Enter SQL Query:", 
                           "SELECT name, email, experience_years FROM resumes ORDER BY experience_years DESC LIMIT 10")
        
        if st.button("Execute Query"):
            try:
                result = db_manager.execute_custom_query(query)
                if result:
                    st.dataframe(pd.DataFrame(result))
                else:
                    st.info("No results returned")
            except Exception as e:
                st.error(f"Query error: {str(e)}")
    else:
        st.info("No resumes in database. Upload some resumes first!")

def analytics_dashboard_page(db_manager):
    st.header("📈 Analytics Dashboard")
    
    resumes = db_manager.get_all_resumes()
    
    if resumes:
        df = pd.DataFrame(resumes)
        
        # Experience distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Experience Distribution")
            if 'experience_years' in df.columns:
                fig = px.histogram(df, x='experience_years', nbins=20, 
                                 title="Distribution of Experience Years")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Skills Word Cloud")
            all_skills = []
            for skills in df['skills']:
                if skills:
                    all_skills.extend(skills.split(','))
            
            if all_skills:
                # Count skill frequency
                from collections import Counter
                skill_counts = Counter(all_skills)
                top_skills = skill_counts.most_common(10)
                
                if top_skills:
                    skills, counts = zip(*top_skills)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(skills, counts)
                    ax.set_xlabel('Frequency')
                    ax.set_title('Top Skills Distribution')
                    plt.tight_layout()
                    st.pyplot(fig)
        
        # Top skills analysis
        st.subheader("🔝 Top Skills Analysis")
        skill_counts = {}
        for skills in df['skills']:
            if skills:
                for skill in skills.split(','):
                    skill = skill.strip().lower()
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        if skill_counts:
            top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
            
            fig = px.bar(skills_df, x='Skill', y='Count', title="Top 10 Skills")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for analytics. Upload some resumes first!")

def ml_predictions_page(ml_models, db_manager):
    st.header("🤖 ML Predictions & Insights")
    
    resumes = db_manager.get_all_resumes()
    
    if resumes and len(resumes) > 5:  # Need minimum data for ML
        df = pd.DataFrame(resumes)
        
        # Train models if not already trained
        if not hasattr(ml_models, 'models_trained'):
            with st.spinner("Training ML models..."):
                ml_models.train_models(df)
                ml_models.models_trained = True
        
        st.subheader("📊 Model Predictions")
        
        # Salary prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Salary Prediction")
            experience = st.slider("Experience Years", 0, 20, 5)
            skills_input = st.text_input("Skills (comma-separated)", "python,machine learning,sql")
            
            if st.button("Predict Salary"):
                predicted_salary = ml_models.predict_salary(experience, skills_input.split(','))
                st.success(f"Predicted Salary: ${predicted_salary:,.2f}")
        
        with col2:
            st.subheader("🎯 Job Role Classification")
            resume_text = st.text_area("Resume Text", height=150)
            
            if st.button("Classify Role") and resume_text:
                predicted_role = ml_models.classify_job_role(resume_text)
                st.success(f"Predicted Role: {predicted_role}")
        
        # Clustering analysis
        st.subheader("🔍 Resume Clustering Analysis")
        if st.button("Perform Clustering Analysis"):
            clusters = ml_models.cluster_resumes(df)
            
            # Visualize clusters
            fig = px.scatter(x=range(len(clusters)), y=clusters, 
                           title="Resume Clusters", labels={'x': 'Resume Index', 'y': 'Cluster'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            cluster_stats = pd.Series(clusters).value_counts().sort_index()
            st.bar_chart(cluster_stats)
    else:
        st.info("Need at least 6 resumes for ML predictions. Upload more resumes!")

def model_performance_page(parser):
    st.header("📏 Model Performance Metrics")
    
    st.subheader("🏷️ NER Model Performance")
    
    # Display NER model metrics
    if hasattr(parser, 'ner_metrics'):
        metrics = parser.ner_metrics
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        with col2:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        with col3:
            st.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")
        
        # Confusion matrix visualization
        if 'confusion_matrix' in metrics:
            st.subheader("Confusion Matrix")
            fig = px.imshow(metrics['confusion_matrix'], 
                          title="NER Model Confusion Matrix",
                          color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    # Test NER model
    st.subheader("🧪 Test NER Model")
    test_text = st.text_area("Enter text to test NER:", 
                           "John Doe is a software engineer at Google with 5 years of experience in Python and machine learning.")
    
    if st.button("Test NER") and test_text:
        entities = parser.extract_entities(test_text)
        
        if entities:
            st.subheader("Extracted Entities:")
            for entity in entities:
                st.write(f"**{entity['text']}** - *{entity['label']}* (confidence: {entity['confidence']:.3f})")
        else:
            st.info("No entities found")
    
    # Relationship extraction performance
    st.subheader("🔗 Relationship Extraction Performance")
    
    sample_relationships = [
        {"subject": "John Doe", "relation": "works_at", "object": "Google", "confidence": 0.95},
        {"subject": "Python", "relation": "skill_of", "object": "John Doe", "confidence": 0.88},
        {"subject": "5 years", "relation": "experience_duration", "object": "software engineering", "confidence": 0.92}
    ]
    
    st.write("Sample extracted relationships:")
    for rel in sample_relationships:
        st.write(f"**{rel['subject']}** → *{rel['relation']}* → **{rel['object']}** (confidence: {rel['confidence']:.2f})")

if __name__ == "__main__":
    main()
