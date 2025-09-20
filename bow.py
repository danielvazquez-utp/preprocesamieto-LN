# Importamos la clase CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Definimos el corpus (la colección de documentos)
corpus = [
    'El gato se sentó en la alfombra.',
    'El perro se sentó en el tronco.'
]

# Creamos una instancia de CountVectorizer
vectorizador = CountVectorizer()

# Convertimos el corpus en una matriz BoW
X = vectorizador.fit_transform(corpus)

# Obtenemos la lista de palabras del vocabulario
palabras_vocabulario = vectorizador.get_feature_names_out()

# Convertimos la matriz en una representación densa para visualización
matriz_bow = X.toarray()

# Imprimimos los resultados para ver la matriz y el vocabulario
print("Vocabulario:", palabras_vocabulario)
print("Matriz BoW:\n", matriz_bow)