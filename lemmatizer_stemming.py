# lematizacion
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos='v'))
print(lemmatizer.lemmatize("better", pos='a'))
print(lemmatizer.lemmatize("cats"))
# stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem("running"))
print(stemmer.stem("better"))
print(stemmer.stem("cats"))