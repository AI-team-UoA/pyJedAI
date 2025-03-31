from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import numpy as np

def vectorFromTFIDF(df, column_name) -> list:          
        column_data = df[column_name].tolist()
        if not isinstance(column_data[0], str):
            column_data = [str(x) for x in column_data]
        column_data = [item for item in column_data if len(item)!=0]
        if len(column_data) == 0:
            return None        

        def custom_tokenizer(text):
            # Tokenize words
            if isinstance(text,str):
                tokens = word_tokenize(text.lower())
            # Remove stopwords
            stop_words = set(stopwords.words('english') + list(string.punctuation))
            filtered_tokens = [token for token in tokens if token not in stop_words]
            # Apply stemming or lemmatization
            stemmer = PorterStemmer()
            # You can choose either stemming or lemmatization here
            # For this example, we'll use lemmatization
            processed_tokens = [stemmer.stem(token) for token in filtered_tokens]
            if len(processed_tokens) == 0:
                return None
            return processed_tokens

        tokenized_data = [custom_tokenizer(x) for x in column_data]
        tokenized_data = [item for item in tokenized_data if item is not None]
            
        if len(tokenized_data) == 0:
                tokenized_data = column_data    
        def preprocessor(t):
            return t
        def tokenizer(t):
            return t 
        # Create TfidfVectorizer object
        vectorizer = TfidfVectorizer(max_features=1000, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform( tokenized_data)
        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        # Calculate TF-IDF scores for each word
        scores = tfidf_matrix.toarray().sum(axis=0)
        # Sort words by score and select top 512
        top_512_words = np.argsort(scores)[::-1][:512]
        top_512_words_list = [feature_names[i] for i in top_512_words]
        # top_512_words_list.append(column_name)
        return top_512_words_list