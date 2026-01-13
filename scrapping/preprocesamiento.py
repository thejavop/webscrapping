import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class PreprocesadorNoticias:
    # Lista de stop words en español (palabras comunes sin valor semántico)
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
    'oct', 'nov', 'dic','efe','abc','ee', 'uu'  # ← Añadido ee y uu por si se escapan
]

    def __init__(self, nombre_archivo='abc_news.csv'):
        self.archivo_entrada = nombre_archivo
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words=self.STOP_WORDS_ES,
            ngram_range=(1, 2),  # Unigramas y bigramas
            min_df=2  # Ignorar palabras que aparecen en menos de 2 documentos
        )
        self.tfidf_matrix = None

    def cargar_datos(self):
        """Carga el CSV y elimina duplicados"""
        print(f"📂 Cargando datos desde: {os.path.abspath(self.archivo_entrada)}")
        try:
            self.df = pd.read_csv(self.archivo_entrada)
            antes = len(self.df)
            self.df = self.df.drop_duplicates(subset=['titulo']).dropna(subset=['categoria'])
            print(f"✓ Cargados {len(self.df)} artículos únicos (Eliminados {antes - len(self.df)} duplicados)")
            print(f"✓ Categorías: {self.df['categoria'].unique().tolist()}")
            print(f"\n📊 Distribución por categoría:")
            print(self.df['categoria'].value_counts())
        except FileNotFoundError:
            print(f"❌ Error: No se encuentra '{self.archivo_entrada}'")
            exit()

    def limpiar_texto(self, texto):
        """Limpia el texto: lowercase, normaliza abreviaturas, quita números y caracteres especiales"""
        if pd.isna(texto) or not isinstance(texto, str):
            return ""

        # Lowercase
        texto = texto.lower()
        
        # ← NUEVO: Normalizar abreviaturas comunes ANTES de limpiar caracteres especiales
        # Esto convierte "EE.UU." → "estadosunidos" (una sola palabra)
        # IMPORTANTE: El orden importa - patrones más específicos primero
        abreviaturas = {
            # Estados Unidos (todas las variantes)
            r'\bee\.?\s?uu\.?\b': 'estadosunidos',           # EE.UU., ee.uu., EEUU
            r'\buu\.?\s?ee\.?\b': 'estadosunidos',           # UU.EE.
            r'\bestados\s+unidos(?:\s+de\s+am[eé]rica)?\b': 'estadosunidos',  # Estados Unidos (de América)
            
            # Recursos Humanos
            r'\brr\.?\s?hh\.?\b': 'recursoshumanos',         # RR.HH.
            r'\brecursos\s+humanos\b': 'recursoshumanos',    # Recursos Humanos
            
            # Unión Europea
            r'\bunion\s+europea\b': 'unioneuropea',          # Unión Europea
            r'\bue\b': 'unioneuropea',                       # UE
            r'\bpresos\s+pol[ií]ticos\b': 'presospoliticos', # Presos políticos
            r'\bderechos\s+humanos\b': 'derechoshumanos',    # Derechos humanos
            r'\bcambio\s+clim[aá]tico\b': 'cambioclimatico', # Cambio climático
            r'\binteligencia\s+artificial\b': 'inteligenciaartificial', # Inteligencia artificial


            # Organizaciones internacionales
            r'\bnaciones\s+unidas\b': 'nacionesunidas',      # Naciones Unidas
            r'\bonu\b': 'nacionesunidas',                    # ONU
            r'\botan\b': 'otan',                             # OTAN
            r'\bfmi\b': 'fmi',                               # FMI
            
            # Países compuestos
            r'\breino\s+unido\b': 'reinounido',              # Reino Unido
            r'\barabia\s+saud[ií]\b': 'arabiasaudi',         # Arabia Saudí
            r'\bcorea\s+del\s+sur\b': 'coreadelsur',         # Corea del Sur
            r'\bcorea\s+del\s+norte\b': 'coreadelnorte',     # Corea del Norte
            r'\bnueva\s+zelanda\b': 'nuevazelanda',          # Nueva Zelanda
            
            # Términos económicos
            r'\bpib\b': 'pib',                               # PIB
            r'\biva\b': 'iva',                               # IVA
        }
        
        for patron, reemplazo in abreviaturas.items():
            texto = re.sub(patron, reemplazo, texto)

        # Quitar números
        texto = re.sub(r'\d+', '', texto)

        # Quitar caracteres especiales (mantener letras españolas y espacios)
        texto = re.sub(r'[^a-záéíóúñü\s]', ' ', texto)

        # Quitar espacios múltiples
        texto = " ".join(texto.split())

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
        
        # ← NUEVO: Reset de índices para evitar desincronización con tfidf_matrix
        self.df = self.df.reset_index(drop=True)
        
        print(f"✓ Texto limpiado (Eliminados {antes - len(self.df)} artículos con texto muy corto)")

        # Guardar CSV limpio
        self.df.to_csv('abc_news_limpio.csv', index=False, encoding='utf-8')
        print(f"✓ CSV limpio guardado: 'abc_news_limpio.csv'")

    def aplicar_tfidf(self):
        """Convierte cada artículo en un vector de números usando TF-IDF"""
        print("\n🔢 Aplicando TF-IDF...")

        # Ajustar y transformar
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['texto_limpio'])

        print(f"✓ TF-IDF aplicado")
        print(f"  - Shape de la matriz: {self.tfidf_matrix.shape}")
        print(f"  - {self.tfidf_matrix.shape[0]} documentos")
        print(f"  - {self.tfidf_matrix.shape[1]} features (palabras/bigramas únicos)")
        print(f"  - Vocabulario total: {len(self.vectorizer.vocabulary_)} términos")

    def dividir_train_test(self, test_size=0.2, random_state=42):
        """Divide los datos en 80% entrenamiento y 20% prueba"""
        print(f"\n✂️ Dividiendo datos en Train/Test ({int((1-test_size)*100)}% / {int(test_size*100)}%)...")

        X = self.tfidf_matrix
        y = self.df['categoria'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Mantiene las proporciones de categorías
        )

        print(f"✓ División completada:")
        print(f"  - Train: {self.X_train.shape[0]} artículos")
        print(f"  - Test: {self.X_test.shape[0]} artículos")

        # Mostrar distribución por categoría
        print(f"\n📊 Distribución en Train:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for cat, count in zip(unique, counts):
            print(f"  - {cat}: {count}")

        print(f"\n📊 Distribución en Test:")
        unique, counts = np.unique(self.y_test, return_counts=True)
        for cat, count in zip(unique, counts):
            print(f"  - {cat}: {count}")

    def palabras_importantes_por_categoria(self, top_n=15):
        """BONUS: Muestra las palabras más importantes de cada categoría"""
        print(f"\n⭐ Top {top_n} palabras más importantes por categoría:")
        print("="*80)

        feature_names = np.array(self.vectorizer.get_feature_names_out())

        for categoria in sorted(self.df['categoria'].unique()):
            print(f"\n🔹 {categoria.upper()}:")

            # Filtrar artículos de esta categoría
            indices = self.df[self.df['categoria'] == categoria].index

            # Sumar TF-IDF de todos los documentos de esta categoría
            categoria_tfidf = self.tfidf_matrix[indices].sum(axis=0).A1

            # Obtener las top N palabras
            top_indices = categoria_tfidf.argsort()[-top_n:][::-1]
            top_palabras = feature_names[top_indices]
            top_scores = categoria_tfidf[top_indices]

            for i, (palabra, score) in enumerate(zip(top_palabras, top_scores), 1):
                print(f"   {i:2d}. {palabra:25s} (score: {score:.2f})")

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
        print("✓ Gráfico guardado: 'grafico_categorias.png'")

    def ejecutar_pipeline_completo(self):
        """Ejecuta el pipeline completo de preprocesamiento"""
        print("="*80)
        print("🚀 INICIANDO PIPELINE DE PREPROCESAMIENTO")
        print("="*80)

        self.cargar_datos()
        self.ejecutar_limpieza()
        self.aplicar_tfidf()
        self.dividir_train_test()
        self.palabras_importantes_por_categoria(top_n=15)
        self.analizar_estadisticas()
        self.generar_grafico()

        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"\n📁 Archivos generados:")
        print(f"  - abc_news_limpio.csv")
        print(f"  - grafico_categorias.png")
        print(f"\n💾 Datos listos para entrenamiento:")
        print(f"  - X_train: {self.X_train.shape}")
        print(f"  - X_test: {self.X_test.shape}")
        print(f"  - y_train: {self.y_train.shape}")
        print(f"  - y_test: {self.y_test.shape}")


if __name__ == "__main__":
    # Crear instancia y ejecutar pipeline completo
    prep = PreprocesadorNoticias('abc_news.csv')
    prep.ejecutar_pipeline_completo()