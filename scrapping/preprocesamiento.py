import matplotlib
matplotlib.use('Agg')  # ← AÑADIR ESTA LÍNEA AL INICIO
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import re
import os
import unicodedata

class PreprocesadorNoticias:
    # Lista de stop words en español
    STOP_WORDS_ES = [
    'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
    'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo',
    'pero', 'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese',
    'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'muy', 'sin', 'vez',
    'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno', 'mismo', 'yo', 'también',
    'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'desde', 'grande', 'eso',
    'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella', 'sí', 'día', 'uno', 'bien',
    'poco', 'deber', 'entonces', 'poner', 'cosa', 'tanto', 'hombre', 'parecer',
    'tan', 'donde', 'ahora', 'parte', 'después', 'vida', 'quedar', 'siempre',
    'creer', 'hablar', 'llevar', 'dejar', 'nada', 'cada', 'seguir', 'menos',
    'nuevo', 'encontrar', 'algo', 'solo', 'estos', 'trabajar', 'cual', 'tres',
    'tal', 'ha', 'han', 'las', 'los', 'una', 'unos', 'unas', 'al', 'del', 'son',
    'es', 'fue', 'fueron', 'era', 'eran', 'siendo', 'sido', 'está', 'están',
    'estaba', 'estaban', 'he', 'has', 'hemos', 'había', 'habían', 'te', 'tu', 'tus',
    'ese', 'esa', 'esos', 'esas', 'este', 'esta', 'estos', 'estas', 'aquel',
    'aquella', 'aquellos', 'aquellas', 'mi', 'mis', 'ti', 'su', 'sus', 'nuestro',
    'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras',
    'ante', 'bajo', 'cabe', 'contra', 'durante', 'mediante', 'salvo', 'según', 'excepto',
    'hacia', 'tras', 'dentro', 'fuera', 'encima', 'debajo', 'delante', 'detrás',
    'quien', 'quienes', 'cuyo', 'cuya', 'cuyos', 'cuyas', 'cuanto', 'cuanta', 'cuantos', 'cuantas',
    'cual', 'cuales', 'cómo', 'dónde', 'cuándo', 'cuánto', 'cuánta', 'cuántos', 'cuántas',
    'ambos', 'ambas', 'varios', 'varias', 'muchos', 'muchas', 'pocos', 'pocas',
    'otros', 'otras', 'algunos', 'algunas', 'ninguno', 'ninguna', 'ningunos', 'ningunas',
    'todavía', 'aún', 'apenas', 'casi', 'solo', 'solamente', 'tampoco', 'todavía',
    'mientras', 'aunque', 'sino', 'mas', 'pues', 'luego', 'conque',
    'allí', 'allá', 'acá', 'aquí', 'ahí', 'arriba', 'abajo', 'cerca', 'lejos',
    'siempre', 'nunca', 'jamás', 'todavía', 'aún', 'pronto', 'tarde', 'temprano',
    'hoy', 'ayer', 'mañana', 'anoche', 'anteanoche', 'anteayer',
    'primero', 'segundo', 'tercero', 'último', 'anterior', 'siguiente', 'próximo',
    'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez', 'cien', 'mil',
    'puede', 'pueden', 'podría', 'podrían', 'debe', 'deben', 'debería', 'deberían',
    'hace', 'hacen', 'hizo', 'hicieron', 'hará', 'harán', 'haría', 'harían',
    'dice', 'dicen', 'dijo', 'dijeron', 'dirá', 'dirán', 'diría', 'dirían',
    'va', 'van', 'voy', 'vas', 'vamos', 'vais', 'iba', 'iban', 'irá', 'irán',
    'sea', 'sean', 'soy', 'eres', 'somos', 'sois', 'seré', 'serás', 'será', 'serán',
    'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo','enero',
    'febrero', 'marzo', 'abril', 'mayo', 'junio','julio', 'agosto', 'septiembre', 'octubre', 
    'noviembre', 'diciembre','ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 
    'oct', 'nov', 'dic','efe','abc','ee', 'uu','asi'
    ]

    def __init__(self, nombre_archivo='abc_news.csv'):
        self.archivo_entrada = nombre_archivo
        self.df = None

    def cargar_datos(self):
        """Carga el CSV y elimina duplicados"""
        print(f"📂 Cargando datos desde: {os.path.abspath(self.archivo_entrada)}")
        try:
            self.df = pd.read_csv(self.archivo_entrada)
            print(f"✓ Cargados {len(self.df)} artículos únicos.")
            print(f"✓ Categorías: {self.df['categoria'].unique().tolist()}")
            print(f"\n📊 Distribución por categoría:")
            print(self.df['categoria'].value_counts())
        except FileNotFoundError:
            print(f"❌ Error: No se encuentra '{self.archivo_entrada}'")
            exit()

    def eliminar_tildes(self, texto):
        """Elimina tildes y acentos de un texto"""
        texto_nfd = unicodedata.normalize('NFD', texto)
        texto_sin_tildes = ''.join(
            char for char in texto_nfd
            if unicodedata.category(char) != 'Mn'
        )
        return texto_sin_tildes

    def limpiar_texto(self, texto):
        """Limpia el texto: lowercase, normaliza abreviaturas, quita números, caracteres especiales, tildes y stopwords"""
        if pd.isna(texto) or not isinstance(texto, str):
            return ""

        # Lowercase
        texto = texto.lower()
        
        # Normalizar abreviaturas comunes ANTES de limpiar caracteres especiales
        abreviaturas = {
            r'\bee\.?\s?uu\.?\b': 'estadosunidos',
            r'\buu\.?\s?ee\.?\b': 'estadosunidos',
            r'\bestados\s+unidos(?:\s+de\s+am[eé]rica)?\b': 'estadosunidos',
            r'\busa\b': 'estadosunidos',
            r'\brr\.?\s?hh\.?\b': 'recursoshumanos',
            r'\brecursos\s+humanos\b': 'recursoshumanos',
            r'\bunion\s+europea\b': 'unioneuropea',
            r'\bue\b': 'unioneuropea',
            r'\bpresos\s+pol[ií]ticos\b': 'presospoliticos',
            r'\bderechos\s+humanos\b': 'derechoshumanos',
            r'\bcambio\s+clim[aá]tico\b': 'cambioclimatico',
            r'\binteligencia\s+artificial\b': 'inteligenciaartificial',
            r'\bnaciones\s+unidas\b': 'nacionesunidas',
            r'\bonu\b': 'nacionesunidas',
            r'\botan\b': 'otan',
            r'\bfmi\b': 'fmi',
            r'\breino\s+unido\b': 'reinounido',
            r'\barabia\s+saud[ií]\b': 'arabiasaudi',
            r'\bcorea\s+del\s+sur\b': 'coreadelsur',
            r'\bcorea\s+del\s+norte\b': 'coreadelnorte',
            r'\bnueva\s+zelanda\b': 'nuevazelanda',
            r'\bpib\b': 'pib',
            r'\biva\b': 'iva',
        }
        
        for patron, reemplazo in abreviaturas.items():
            texto = re.sub(patron, reemplazo, texto)

        # Eliminar tildes
        texto = self.eliminar_tildes(texto)

        # Quitar números
        texto = re.sub(r'\d+', '', texto)

        # Quitar caracteres especiales (mantener solo letras sin tilde y espacios)
        texto = re.sub(r'[^a-zn\s]', ' ', texto)

        # Quitar espacios múltiples
        texto = " ".join(texto.split())

        # Eliminar stopwords
        palabras = texto.split()
        palabras_filtradas = [p for p in palabras if p not in self.STOP_WORDS_ES]
        texto = " ".join(palabras_filtradas)

        return texto

    def ejecutar_limpieza(self):
        """Combina título + descripción y limpia el texto"""
        print("\n🧹 Limpiando texto...")

        # Combinar título + descripción
        self.df['texto_completo'] = (
            self.df['titulo'].fillna('') + " " +
            self.df['descripcion'].fillna('')
        )

        # Aplicar limpieza
        self.df['texto_limpio'] = self.df['texto_completo'].apply(self.limpiar_texto)

        # Eliminar textos vacíos o muy cortos
        antes = len(self.df)
        self.df = self.df[self.df['texto_limpio'].str.len() > 10]
        
        # Reset de índices
        self.df = self.df.reset_index(drop=True)
        
        print(f"✓ Texto limpiado (Eliminados {antes - len(self.df)} artículos con texto muy corto)")
        
        # Mostrar ejemplos
        print(f"\n📝 Ejemplo de limpieza:")
        print(f"  ANTES: {self.df['texto_completo'].iloc[0][:100]}...")
        print(f"  DESPUÉS: {self.df['texto_limpio'].iloc[0][:100]}...")

        # Guardar CSV limpio
        self.df.to_csv('abc_news_limpio.csv', index=False, encoding='utf-8')
        print(f"\n✓ CSV limpio guardado: 'abc_news_limpio.csv'")

    def analizar_estadisticas(self):
        """Muestra estadísticas básicas del dataset"""
        print("\n📊 Estadísticas del dataset:")

        # Palabras por título
        palabras_titulo = self.df['titulo'].fillna('').apply(lambda x: len(x.split())).to_numpy()
        print(f"\nPalabras por título:")
        print(f"  - Media: {np.mean(palabras_titulo):.2f}")
        print(f"  - Desviación estándar: {np.std(palabras_titulo):.2f}")
        print(f"  - Min: {np.min(palabras_titulo)}")
        print(f"  - Max: {np.max(palabras_titulo)}")

        # Palabras por texto limpio
        palabras_texto = self.df['texto_limpio'].apply(lambda x: len(x.split())).to_numpy()
        print(f"\nPalabras por texto limpio (título + descripción):")
        print(f"  - Media: {np.mean(palabras_texto):.2f}")
        print(f"  - Desviación estándar: {np.std(palabras_texto):.2f}")
        print(f"  - Min: {np.min(palabras_texto)}")
        print(f"  - Max: {np.max(palabras_texto)}")

    def generar_grafico(self):
        """Genera gráfico de distribución por categoría"""
        print("\n📈 Generando gráfico...")

        counts = self.df['categoria'].value_counts()

        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title('Distribución de Artículos por Categoría', fontsize=14, fontweight='bold')
        plt.xlabel('Categoría', fontsize=12)
        plt.ylabel('Número de Artículos', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('grafico_categorias.png', dpi=300)
        plt.close()  # ← IMPORTANTE: Cerrar la figura
        print("✓ Gráfico guardado: 'grafico_categorias.png'")

    def ejecutar_pipeline_completo(self):
        """Ejecuta el pipeline completo de preprocesamiento"""
        print("="*80)
        print("🚀 INICIANDO PIPELINE DE PREPROCESAMIENTO")
        print("="*80)

        self.cargar_datos()
        self.ejecutar_limpieza()
        self.analizar_estadisticas()
        self.generar_grafico()

        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"\n📁 Archivos generados:")
        print(f"  - abc_news_limpio.csv")
        print(f"  - grafico_categorias.png")
        print(f"\n💾 Datos listos para entrenamiento con TF-IDF")


if __name__ == "__main__":
    prep = PreprocesadorNoticias('abc_news.csv')
    prep.ejecutar_pipeline_completo()