# Importamos las clases necesarias de la biblioteca gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# Descargamos los modelos de tokenización de NLTK (solo la primera vez)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Definimos el corpus (la colección de documentos)
corpus = """
El gato se sentó en la alfombra.
El perro se sentó en el tronco.
El gato y el perro son amigos.
Los animales son muy inteligentes.
El sol brilla en el cielo.
La luna es visible por la noche.
"""

# 1. Preprocesamiento: Tokenizamos las oraciones y las palabras
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sent_tokenize(corpus)]

# 2. Entrenamos el modelo Word2Vec
# size: dimensión del vector (longitud)
# window: distancia máxima entre la palabra actual y las predichas
# min_count: ignora todas las palabras con frecuencia total menor que este valor
model = Word2Vec(sentences=tokenized_sentences, vector_size=10, window=5, min_count=1)

# 3. Exploramos el modelo
# Obtenemos el vector (embedding) de una palabra
vector_perro = model.wv['perro']
print(f"Vector para la palabra 'perro':\n{vector_perro}\n")

# Encontramos las 3 palabras más similares a 'gato'
palabras_similares_a_gato = model.wv.most_similar('gato', topn=3)
print(f"Palabras más similares a 'gato':\n{palabras_similares_a_gato}")