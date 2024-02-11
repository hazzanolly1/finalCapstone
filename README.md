## Table of contents
1. Introduction
2. Features
3. Dependencies
4.  Installation
5. Usage
6. Inputs
7. Outputs
8. Contributors

# Sentiment Analysis

1.0 Introduction
This repository highlights sentiment analysis performed on a dataset amazon_products_reviews.csv containing product reviews using Natural Language Processing (NLP) techniques. It combines the capabilities of two libraries, spaCy and NLTK, to analyze the sentiment of the reviews.


## 2.0 Features
1. Utilizes spaCy and NLTK libraries for sentiment analysis.
2. Performs sentiment analysis using two methods: spaCyTextBlob and VADER.
3. Generates a comprehensive sentiment analysis report in PDF format. 

## 3.0 Dependencies
Python 3.x
pandas
spacy
spacytextblob
nltk
fpdf

## 4.0 Installation
1. Install Python (if not already installed) from Python's official website.
2. Install the required Python libraries using pip:
pip install pandas spacy nltk fpdf spacytextblob
3. Download the spaCy English language model:
python -m spacy download en_core_web_sm
4. Download the VADER lexicon:
import nltk
nltk.download('vader_lexicon')

    
## 5.0 Usage
1. Clone the repository https://github.com/hazzanolly1/finalCapstone
2. Ensure that the dataset (amazon_product_reviews.csv) is available in relevant path  on your  system or adjust the path as suitable for you.
3. Run the script using Python:
python sentiment_analysis.py
4. Follow the on-screen instructions on your computer to perform sentiment analysis on the dataset amazon_product_reviews.csv.
5. Upon completion, a sentiment analysis report will be generated in PDF format.

## 6.0 Inputs
The dataset file (amazon_product_reviews.csv) containing product reviews.
Parameters for sentiment analysis such as Spacy,TextBlob and NLTK libraries.
## 7.0 Outputs
The sentiment analysis results printed for each review in the dataset.
A PDF report summarizing the sentiment analysis process, dataset description, preprocessing steps, sentiment analysis results, evaluation of results, and insights into the model's strengths and limitations.
## 8.0 Contributions
Contributions are always welcome!
Please adhere to this project's `code of conduct
