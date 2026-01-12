import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class PreprocesadorNoticias:
    def __init__(self, nombre_archivo='bbc_news.csv'):
        self.archivo_entrada = nombre_archivo
        self.df = None
        self.vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')

    def cargar_datos(self):
        print(f" Cargando datos desde: {os.path.abspath(self.archivo_entrada)} ")
        try:
            self.df = pd.read_csv(self.archivo_entrada, dtype=str)
            antes = len(self.df)
            self.df = self.df.drop_duplicates(subset=['titulo']).dropna(subset=['categoria'])
            print(f"Cargados {len(self.df)} artículos únicos (Borrados {antes - len(self.df)}).")
        except FileNotFoundError:
            print(f"Error: No se encuentra '{self.archivo_entrada}'. Asegúrate de estar en la carpeta correcta.")
            exit()

    def limpiar_texto_extremo(self, texto):
        """Borra todo lo que no sean letras minúsculas"""
        if pd.isna(texto) or not isinstance(texto, str): return ""
        texto = texto.lower()
        # reemplaza cualquier símbolo o número por un espacio
        texto = re.sub(r'[^a-z\s]', ' ', texto)
        return " ".join(texto.split())

    def ejecutar_limpieza(self):
        print(" Procesando texto para la columna de IA ")
        # Unimos título y descripción y limpiamos
        cuerpo = self.df['titulo'].fillna('') + " " + self.df['descripcion'].fillna('')
        self.df['texto_final'] = cuerpo.apply(self.limpiar_texto_extremo)
        # creamos el nuevo csv limpio 
        self.df.to_csv('bbc_news_limpio.csv', index=False, encoding='utf-8')
        print("Bien hecho, 'bbc_news_limpio.csv' generado correctamente.")

    def analizar_estadisticas(self):
        print("\n Estadísticas Numpy ")
        palabras = self.df['titulo'].fillna('').apply(lambda x: len(x.split())).to_numpy()
        print(f"Media de palabras: {np.mean(palabras):.2f} | Desviación: {np.std(palabras):.2f}")

    def generar_grafico(self):
        print("\nGenerando Gráfico Matplotlib ")
        counts = self.df['categoria'].value_counts()
        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar', color='firebrick', edgecolor='black')
        plt.title('Noticias por Categoría')
        plt.tight_layout()
        plt.savefig('grafico_categorias.png')
        print("Gráfico guardado.")

if __name__ == "__main__":
    pre = PreprocesadorNoticias()
    pre.cargar_datos()
    pre.analizar_estadisticas()
    pre.ejecutar_limpieza()
    pre.generar_grafico()