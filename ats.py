import re
from nltk.corpus import stopwords

# --- Setup ---
stop_words = set(stopwords.words('english'))

def get_keywords(text):
    """Extracts and cleans keywords from a given text."""
    # Simple cleaning: lowercase, remove non-alphanumeric, split into words
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = set(text.split())
    
    # Remove stop words
    keywords = words - stop_words
    return keywords

def calculate_ats_score(resume_text, job_description_text):
    """Calculates the ATS score based on keyword matching."""
    
    # Extract keywords from both texts
    resume_keywords = get_keywords(resume_text)
    jd_keywords = get_keywords(job_description_text)
    
    # Find the matching keywords
    matching_keywords = resume_keywords.intersection(jd_keywords)
    
    # Calculate the score
    if not jd_keywords:
        return 0.0 # Avoid division by zero
        
    score = (len(matching_keywords) / len(jd_keywords)) * 100
    
    # Return the score and the list of matched/missing keywords for feedback
    missing_keywords = jd_keywords - resume_keywords
    
    return {
        'score': round(score, 2),
        'matching_keywords': sorted(list(matching_keywords)),
        'missing_keywords': sorted(list(missing_keywords))
    }

# --- Example Usage ---
if __name__ == '__main__':
    sample_resume = """
    John Doe | Data Analyst | john.doe@email.com
    Experience with Python, Pandas, and SQL to analyze complex datasets.
    Created dashboards in Tableau to visualize key performance indicators.
    Expert in statistical analysis and data modeling.
    """
    
    sample_jd = """
    We are looking for a Data Analyst with strong SQL and Python skills.
    The ideal candidate will have experience with Tableau for creating data visualizations.
    Knowledge of machine learning is a plus.
    """
    
    ats_results = calculate_ats_score(sample_resume, sample_jd)
    
    print(f"ATS Score: {ats_results['score']}%")
    print(f"\nMatching Keywords: {ats_results['matching_keywords']}")
    print(f"\nMissing Keywords: {ats_results['missing_keywords']}")