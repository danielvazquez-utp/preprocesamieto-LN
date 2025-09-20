# Importamos la clase TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Definimos el corpus (la colección de documentos)
corpus = [
    'El gato se sentó en la alfombra.',
    'El perro se sentó en el tronco.',
    'El gato y el perro son amigos.'
]

# 1. Creamos una instancia de TfidfVectorizer
vectorizador_tfidf = TfidfVectorizer()

# 2. Convertimos el corpus en una matriz TF-IDF
X_tfidf = vectorizador_tfidf.fit_transform(corpus)

# 3. Obtenemos el vocabulario de las palabras
palabras_vocabulario = vectorizador_tfidf.get_feature_names_out()

# 4. Convertimos la matriz en una representación densa para visualización
matriz_tfidf = X_tfidf.toarray()

# 5. Imprimimos los resultados
import pandas as pd

# Creamos un DataFrame para una mejor visualización
df_tfidf = pd.DataFrame(matriz_tfidf, columns=palabras_vocabulario, index=[f'Doc {i+1}' for i in range(len(corpus))])

print("Vocabulario:", palabras_vocabulario)
print("\nMatriz TF-IDF:\n", df_tfidf.round(2))