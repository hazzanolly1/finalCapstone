import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fpdf import FPDF
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Add spaCyTextBlob component to the pipeline
spacy_text_blob = SpacyTextBlob(nlp)

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to analyze sentiment of a product review using VADER and spaCyTextBlob
def sentiment_analysis(text):
    # Analyze sentiment using spaCyTextBlob
    doc = nlp(text)
    spacytextblob_polarity = doc._.polarity if doc._.polarity is not None else 0
    
    # Analyze sentiment using VADER
    vader_scores = sid.polarity_scores(text)
    vader_polarity = vader_scores['compound']
    
    # Combine polarity scores from spaCyTextBlob and VADER
    combined_polarity = (spacytextblob_polarity + vader_polarity) / 2
    
    # Adjusted thresholds for sentiment classification
    positive_threshold = 0.3
    negative_threshold = -0.3
    
    # Determine sentiment label based on combined polarity score and adjusted thresholds
    if combined_polarity > positive_threshold:
        sentiment = 'positive'
    elif combined_polarity < negative_threshold:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {'polarity': combined_polarity, 'sentiment': sentiment}


# Load the dataset
product_dataset = pd.read_csv('C:/Users/LENOVO/Desktop/AI +Data Analytics Certifications Learning/HyperionDev Data Science Python Folder/Task 21 - Capstone Project - NLP Applications/amazon_product_reviews.csv', encoding='ISO-8859-1')

# Remove missing inputs
cleaned_data_input = product_dataset.dropna(subset=['reviews.text'])

# Apply sentiment analysis to the 'reviews.text' column
cleaned_data_input['sentiment_analysis'] = cleaned_data_input['reviews.text'].apply(sentiment_analysis)

# Sample product reviews for testing
sample_reviews = cleaned_data_input['reviews.text'].sample(n=5, random_state=42)

# Print sentiment analysis results for each review
for index, review in sample_reviews.iteritems():
    result = sentiment_analysis(review)
    print(f'Review #{index}:\n{review}\nSentiment Analysis Result: {result}\n')

# Generate a brief report in PDF file
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Sentiment Analysis Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()

# Report content
pdf.chapter_title('1. Dataset Description')
pdf.chapter_body('The dataset used for sentiment analysis is "amazon_product_reviews.csv".')

pdf.chapter_title('2. Preprocessing Steps')
pdf.chapter_body('The "review.text" column was selected, and missing values were removed using the dropna() function.')

pdf.chapter_title('3. Sentiment Analysis Results')
for index, review in sample_reviews.iteritems():
    result = sentiment_analysis(review)
    pdf.chapter_body(f'Review #{index}:\n{review}\nSentiment Analysis Result: {result}\n')

pdf.chapter_title('4. Evaluation of Results')
pdf.chapter_body('The results indicate the sentiment polarity and sentiment label for each sample review.')

pdf.chapter_title('5. Insights into Model\'s Strengths and Limitations')
pdf.chapter_body('Both VADER and spaCyTextBlob were used for sentiment analysis. They provide sentiment polarity and labels, allowing for a nuanced understanding of sentiment.')

# Save the PDF report
pdf.output('sentiment_analysis_report.pdf')
