import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
#from nltk.

text = "This is Andrew's text, isn't it?"

tokenizer = nltk.tokenize.WhitespaceTokenizer()
tokenizer.tokenize(text)
tk = tokenizer.tokenize(text)




tokeniza = nltk.tokenize.TreebankWordTokenizer()

tki = tokeniza.tokenize(text)


token = nltk.tokenize.WordPunctTokenizer()
tkz = token.tokenize(text)

 


word = "feet cats wolves talked"
tk =nltk.tokenize.TreebankWordTokenizer()
tokens = tk.tokenize(text)

stemmer = nltk.stem.PorterStemmer()
" ".join(stemmer.stem(token) for token in tk )



stemma = nltk.stem.WordNetLemmatizer()
" ".join(stemma.lemmatize(token) for token in tk)



####PYTHON TF-IDF example

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

texts = ["good movie", "not a good movie", "did not like", 
         "i like it", "good one"]

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range = (1, 2))
features = tfidf.fit_transform(texts)

pd.DataFrame(features.todense(),
             columns = tfidf.get_feature_names())


from sklearn.feature_extraction.text import Ha


































import nltk
nltk.download()


























