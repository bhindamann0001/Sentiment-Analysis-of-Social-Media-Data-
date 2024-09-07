#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install textblob


# ## **Importing libraries**

# In[2]:


import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.metrics import classification_report
from skimpy import skim


# In[3]:


# Load the JSON file into a DataFrame
with open('data.json', 'r') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
df.head(5)


# ## **EDA**

# In[4]:


skim(df)


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# In[7]:


#Distribution of the number of reviews across products
reviews_per_product = df['asin'].value_counts()
plt.figure(figsize=(10, 6))
plt.hist(reviews_per_product, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Number of Reviews Across Products')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Products')
plt.grid(True)
plt.show()



# In[8]:


plt.figure(figsize=(12, 6))
reviews_per_product.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Reviews Across Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=90)  
plt.grid(axis='y')  
plt.tight_layout()  
plt.show()


# In[9]:


# Distribution of reviews per user
reviews_per_user = df['reviewerID'].value_counts()

plt.figure(figsize=(10, 6))
plt.hist(reviews_per_user, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Reviews per User')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Users')
plt.grid(True)
plt.show()


# In[10]:


# Analyze review lengths with boxplot
df['review_length'] = df['reviewText'].str.len()
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['review_length'])
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.show()


# In[11]:


# Calculate descriptive statistics for review lengths
review_length_stats = df['review_length'].describe()

print("Review Length Statistics:")
print(review_length_stats)


# In[12]:


# Plot histogram of review lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['review_length'], kde=True, color='skyblue')
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length (characters)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[13]:


columns_of_interest = ['overall', 'verified',  'vote']

for column in columns_of_interest:
    unique_values = df[column].unique()
    sum_unique_values = len(unique_values)
    print(f"Unique values in the '{column}' column:")
    print(unique_values)
    print(f"Sum of unique values in the '{column}' column: {sum_unique_values}")
    print()


# In[14]:


#duplicate review
duplicate_reviews = df[df.duplicated(subset=['reviewerID', 'asin', 'reviewText'], keep=False)]
len(duplicate_reviews)


# ## **TEXT BASIC PRE-PROCESSING**

# In[15]:


# Define a function to label the data 
def label_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

# create a new column 'sentiment' based on the rating
df['sentiment_label'] = df['overall'].apply(label_sentiment)

# Checking the changes 
display(df.head())


# In[16]:


df_bkp = df.copy()
# Dropping not usefull columns 
df.drop(columns=['overall', 'reviewerID','asin','style','reviewerName',
                        'unixReviewTime','vote', 'image'], inplace=True)

df.columns


# In[17]:


df = df[df['review_length'] <= 200]
df


# In[18]:


# Plot histogram of review lengths
plt.figure(figsize=(10, 6))
sns.histplot(df['review_length'], kde=True, color='skyblue')
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length (characters)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[19]:


num_rows = len(df)
print("Number of rows in the DataFrame:", num_rows)


# In[20]:


# Print the first 10 contents of the reviewText column
for review_text in df['reviewText'].head(10):
    print(review_text)


# In[21]:


text_data = df['reviewText']


# In[22]:


# Define a function to perform all pre-processing steps
def preprocess_text(text):
    # Lowercasing
    text_lower = text.lower()
    
    # Removing Punctuation
    text_no_punct = text_lower.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    text_tokens = word_tokenize(text_no_punct)
    
    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    text_no_stopwords = [word for word in text_tokens if word not in stop_words]
    
    # Combine tokens into a single string
    processed_text = ' '.join(text_no_stopwords)
    
    return processed_text


# In[23]:


# Apply the pre-processing function to create the new column
df['processedText'] = text_data.apply(preprocess_text)


# In[24]:


df.head()


# In[25]:


df.drop(columns=['verified', 'reviewTime','summary','review_length','reviewText'], inplace=True)

df.columns


# In[26]:


# Randomly select 1000 reviews from the dataset
df_sampled = df.sample(n=1000, random_state=26) 

df_sampled


# ## **MODELING (VADER AND TEXTBLOB)**

# In[27]:


# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Function to apply VADER sentiment analysis
def analyze_sentiment_vader(text):
    sentiment_score = vader_analyzer.polarity_scores(text)
    # Classify sentiment based on compound score
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


# In[28]:


# Function to apply TextBlob sentiment analysis
def analyze_sentiment_textblob(text):
    sentiment = TextBlob(text).sentiment.polarity
    # Classify sentiment based on polarity score
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'


# In[29]:


# Apply sentiment analysis to the pre-processed text data
df_sampled['sentiment_vader'] = df_sampled['processedText'].apply(analyze_sentiment_vader)
df_sampled['sentiment_textblob'] = df_sampled['processedText'].apply(analyze_sentiment_textblob)


# In[30]:


display(df_sampled[['processedText', 'sentiment_label', 'sentiment_vader', 'sentiment_textblob']])


# ## **VALIDATION**

# In[31]:


# Generate classification reports for both models
vader_report = classification_report(df_sampled['sentiment_label'], df_sampled['sentiment_vader'], output_dict=True)
textblob_report = classification_report(df_sampled['sentiment_label'], df_sampled['sentiment_textblob'], output_dict=True)

# Display comparison table
comparison_table = pd.DataFrame({
    "": ['Precision', 'Recall', 'F1-Score'],
    "VADER Positive": [vader_report['Positive']['precision'], vader_report['Positive']['recall'], vader_report['Positive']['f1-score']],
    "VADER Negative": [vader_report['Negative']['precision'], vader_report['Negative']['recall'], vader_report['Negative']['f1-score']],
    "VADER Neutral": [vader_report['Neutral']['precision'], vader_report['Neutral']['recall'], vader_report['Neutral']['f1-score']],
    "TextBlob Positive": [textblob_report['Positive']['precision'], textblob_report['Positive']['recall'], textblob_report['Positive']['f1-score']],
    "TextBlob Negative": [textblob_report['Negative']['precision'], textblob_report['Negative']['recall'], textblob_report['Negative']['f1-score']],
    "TextBlob Neutral": [textblob_report['Neutral']['precision'], textblob_report['Neutral']['recall'], textblob_report['Neutral']['f1-score']]
})

print("Comparison of Sentiment Analysis Models:")
print(comparison_table)


# In[ ]:




