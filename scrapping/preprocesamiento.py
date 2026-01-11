import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar datos
df = pd.read_csv('bbc_news.csv')
print(f"Loaded {len(df)} articles")

# Verificar distribución
print("\nCategory distribution:")
print(df['categoria'].value_counts())

# Eliminar duplicados
df = df.drop_duplicates(subset=['titulo'])
print(f"\nAfter removing duplicates: {len(df)} articles")

# Combinar título + descripción
df['texto_completo'] = df['titulo'] + ' ' + df['descripcion'].fillna('')

# Verificar nulos
print("\nNull values:")
print(df.isnull().sum())

# Guardar datos limpios
df.to_csv('bbc_news_limpio.csv', index=False)
print("\nCleaned data saved to 'bbc_news_limpio.csv'")

# Configurar vectorizador TF-IDF
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.8
)

# Vectorizar textos
X = vectorizer.fit_transform(df['texto_completo'])
y = df['categoria']

print(f"\nMatrix shape: {X.shape}")
print(f"({X.shape[0]} documents × {X.shape[1]} features)")

print("\nTarget distribution:")
print(y.value_counts())
