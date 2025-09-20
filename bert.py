# Importamos la clase principal de la biblioteca
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Cargamos un modelo pre-entrenado de Sentence-BERT
# 'distilbert-base-nli-stsb-mean-tokens' es un modelo optimizado
# para comparar frases.
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# 2. Definimos las frases que queremos codificar
sentences = [
    "El gato se sentó en la alfombra.",
    "Un felino se posó sobre la estera.",
    "El perro está durmiendo en el jardín.",
    "La comida está lista para servir."
]

# 3. Obtenemos los embeddings de las frases
# El método .encode() convierte cada frase en un vector numérico
embeddings = model.encode(sentences)

# Imprimimos la forma de la matriz de embeddings para ver sus dimensiones
print(f"Forma de los embeddings: {embeddings.shape}")

# 4. Calculamos la similitud de coseno entre los embeddings
# Esto nos ayuda a ver qué frases son semánticamente más cercanas
similitudes = cosine_similarity(embeddings)

# Imprimimos la matriz de similitud
print("\nMatriz de similitud de coseno:")
# Redondeamos los valores para una mejor visualización
similitudes_redondeadas = np.round(similitudes, 2)
print(similitudes_redondeadas)

# 5. Analizamos un par de ejemplos específicos
similitud_ejemplo = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"\nSimilitud entre 'El gato se sentó...' y 'Un felino se posó...': {similitud_ejemplo[0][0]:.2f}")

similitud_ejemplo2 = cosine_similarity([embeddings[0]], [embeddings[3]])
print(f"Similitud entre 'El gato se sentó...' y 'La comida está lista...': {similitud_ejemplo2[0][0]:.2f}")