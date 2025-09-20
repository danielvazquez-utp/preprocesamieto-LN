import nltk
# nltk.download()
from nltk.tokenize import word_tokenize, sent_tokenize
# tokenizacion por palabras
text = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
tokens = word_tokenize(text)
print(tokens)
# tokenizacion por oraciones
sentences = sent_tokenize(text)
print(sentences)
# tokenizacion con bert
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Hello Mr. Smith, how are you doing today?"
tokens = tokenizer.tokenize(text)
print(tokens)