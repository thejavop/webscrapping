import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings
import time
import os

warnings.filterwarnings('ignore')


class ClasificadorNoticias:
    """
    Clasificador de noticias usando TF-IDF + Random Forest + Logistic Regression
    
    Mejoras implementadas:
    - Carga datasets preprocesados existentes (no genera nuevos)
    - Eliminado Random Forest optimizado (GridSearchCV)
    - Comparación directa RF base vs Logistic Regression
    - Validación cruzada para ambos modelos
    - Visualizaciones mejoradas
    """

    def __init__(self):
        """Inicializa el clasificador"""
        self.modelo_rf = None
        self.modelo_lr = None
        self.vectorizer = None
        self.categorias = None
        self.tiempo_entrenamiento = {}
        self.metricas_todas = {}
        
        # Datos
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.indices_train = None  # Índices originales del train
        self.indices_test = None   # Índices originales del test
        self.df = None

    # ------------------------------------------------------------------
    # CARGA DE DATOS PREPROCESADOS
    # ------------------------------------------------------------------
    def cargar_datos_preprocesados(self, archivo_csv='abc_news_limpio.csv'):
        """
        Carga el CSV ya preprocesado y reconstruye las matrices TF-IDF
        
        Args:
            archivo_csv: Ruta al CSV limpio generado por preprocesamiento.py
        """
        print("="*80)
        print("📂 CARGANDO DATOS PREPROCESADOS")
        print("="*80)
        
        # Verificar que existe el archivo
        if not os.path.exists(archivo_csv):
            raise FileNotFoundError(
                f"❌ No se encuentra '{archivo_csv}'.\n"
                f"   Debes ejecutar primero 'preprocesamiento.py' para generar el CSV limpio."
            )
        
        print(f"\n📄 Cargando: {os.path.abspath(archivo_csv)}")
        self.df = pd.read_csv(archivo_csv)
        
        print(f"✅ CSV cargado: {len(self.df)} artículos")
        print(f"   Categorías encontradas: {self.df['categoria'].unique().tolist()}")
        
        # Verificar columnas necesarias
        columnas_requeridas = ['texto_limpio', 'categoria']
        if not all(col in self.df.columns for col in columnas_requeridas):
            raise ValueError(
                f"❌ El CSV debe contener las columnas: {columnas_requeridas}\n"
                f"   Columnas encontradas: {self.df.columns.tolist()}"
            )
        
        print(f"\n📊 Distribución por categoría:")
        print(self.df['categoria'].value_counts())
        
        return self.df

    def aplicar_tfidf_y_dividir(self, test_size=0.2, random_state=42):

        print("\n" + "="*80)
        print("🔢 APLICANDO TF-IDF Y DIVIDIENDO DATOS")
        print("="*80)

        from sklearn.model_selection import train_test_split

        # ================================
        # SELECCIÓN EXPLÍCITA DE COLUMNAS
        # ================================
        X_texto = self.df.iloc[:, -2]   # texto_limpio
        y = self.df.iloc[:, -1].values  # categoria
        indices = self.df.index.values

        print("\n📌 Columnas usadas:")
        print(f"   - Texto (X): {self.df.columns[-2]}")
        print(f"   - Etiqueta (y): {self.df.columns[-1]}")

        # ================================
        # TF-IDF (SIN LIMPIEZA ADICIONAL)
        # ================================
        print("\n⚙️  Configurando TF-IDF Vectorizer:")
        print("   - max_features: 1500")
        print("   - ngram_range: (1, 2)")
        print("   - min_df: 2")
        print("   - stop_words: None (ya preprocesado)")

        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            min_df=2
        )

        print("\n⏳ Transformando textos a vectores TF-IDF...")
        X = self.vectorizer.fit_transform(X_texto)

        print(f"✅ TF-IDF aplicado")
        print(f"   - Matriz shape: {X.shape}")

        # ================================
        # TRAIN / TEST SPLIT
        # ================================
        print(f"\n✂️  Dividiendo datos ({int((1-test_size)*100)}% train / {int(test_size*100)}% test)...")

        self.X_train, self.X_test, self.y_train, self.y_test, self.indices_train, self.indices_test = train_test_split(
            X,
            y,
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        self.categorias = sorted(np.unique(self.y_train))

        print(f"✅ División completada:")
        print(f"   - Train: {self.X_train.shape[0]} artículos")
        print(f"   - Test:  {self.X_test.shape[0]} artículos")

        print(f"\n📊 Distribución en Train:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for cat, count in zip(unique, counts):
            print(f"   - {cat}: {count}")



    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------
    def entrenar_random_forest(self):
        """Entrena Random Forest con parámetros por defecto"""
        print("\n" + "="*80)
        print("🌲 ENTRENANDO RANDOM FOREST")
        print("="*80)
        print("Parámetros: n_estimators=100, random_state=42")

        self.modelo_rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        print("\n⏳ Entrenando...")
        inicio = time.time()
        self.modelo_rf.fit(self.X_train, self.y_train)
        tiempo = time.time() - inicio
        self.tiempo_entrenamiento['Random Forest'] = tiempo

        print(f"✅ Random Forest entrenado en {tiempo:.2f} segundos")

    def entrenar_logistic_regression(self):
        """Entrena Logistic Regression"""
        print("\n" + "="*80)
        print("📊 ENTRENANDO LOGISTIC REGRESSION")
        print("="*80)
        print("Parámetros: max_iter=1000, solver='lbfgs'")

        self.modelo_lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            verbose=0
        )

        print("\n⏳ Entrenando...")
        inicio = time.time()
        self.modelo_lr.fit(self.X_train, self.y_train)
        tiempo = time.time() - inicio
        self.tiempo_entrenamiento['Logistic Regression'] = tiempo

        print(f"✅ Logistic Regression entrenado en {tiempo:.2f} segundos")

    # ------------------------------------------------------------------
    # EVALUACIÓN
    # ------------------------------------------------------------------
    def evaluar_modelo(self, modelo, nombre):
        """Evalúa el modelo y retorna métricas completas"""
        print("\n" + "="*80)
        print(f"📊 EVALUANDO {nombre.upper()}")
        print("="*80)

        # Predicciones
        y_pred_train = modelo.predict(self.X_train)
        y_pred_test = modelo.predict(self.X_test)

        # Accuracy
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)

        print(f"\n🎯 ACCURACY:")
        print(f"   - Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   - Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")

        # Reporte de clasificación
        print(f"\n📋 REPORTE DE CLASIFICACIÓN (Test):")
        print("-"*80)
        print(classification_report(
            self.y_test, 
            y_pred_test, 
            target_names=self.categorias,
            digits=4
        ))

        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred_test, labels=self.categorias)

        # Métricas promedio
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, 
            y_pred_test, 
            average='weighted'
        )

        tiempo = self.tiempo_entrenamiento.get(nombre, 0)
        print(f"\n⏱️  Tiempo de entrenamiento: {tiempo:.2f} segundos")

        metricas = {
            'nombre': nombre,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_pred_test': y_pred_test,
            'tiempo_entrenamiento': tiempo
        }

        self.metricas_todas[nombre] = metricas
        return metricas

    def validacion_cruzada(self, modelo, nombre):
        """Realiza validación cruzada de 5-fold"""
        print(f"\n{'='*80}")
        print(f"🔄 VALIDACIÓN CRUZADA (5-FOLD) - {nombre}")
        print(f"{'='*80}")
        
        scores = cross_val_score(
            modelo, 
            self.X_train, 
            self.y_train, 
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        print(f"\n📊 Scores por Fold:")
        for i, score in enumerate(scores, 1):
            print(f"   Fold {i}: {score:.4f} ({score*100:.2f}%)")
        
        print(f"\n📈 RESUMEN:")
        print(f"   - Media:               {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
        print(f"   - Desviación estándar: {scores.std():.4f}")
        print(f"   - Intervalo 95%:       [{scores.mean() - 2*scores.std():.4f}, {scores.mean() + 2*scores.std():.4f}]")
        
        return scores

    # ------------------------------------------------------------------
    # VISUALIZACIONES
    # ------------------------------------------------------------------
    def visualizar_matriz_confusion(self, metricas, guardar=True):
        """Genera heatmap de la matriz de confusión"""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            metricas['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.categorias,
            yticklabels=self.categorias,
            cbar_kws={'label': 'Número de predicciones'}
        )
        
        plt.title(
            f"Matriz de Confusión - {metricas['nombre']}\n"
            f"Accuracy: {metricas['test_accuracy']:.4f}",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        plt.ylabel('Categoría Real', fontsize=12, fontweight='bold')
        plt.xlabel('Categoría Predicha', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if guardar:
            filename = f"matriz_confusion_{metricas['nombre'].lower().replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   ✅ Guardado: {filename}")
        
        plt.close()

    def visualizar_metricas_por_categoria(self, metricas, guardar=True):
        """Genera gráfico de barras con métricas por categoría"""
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test,
            metricas['y_pred_test'],
            labels=self.categorias,
            average=None
        )
        
        df_metricas = pd.DataFrame({
            'Categoría': self.categorias,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.categorias))
        width = 0.25
        
        bars1 = ax.bar(x - width, df_metricas['Precision'], width, label='Precision', color='skyblue')
        bars2 = ax.bar(x, df_metricas['Recall'], width, label='Recall', color='lightcoral')
        bars3 = ax.bar(x + width, df_metricas['F1-Score'], width, label='F1-Score', color='lightgreen')
        
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        ax.set_xlabel('Categoría', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Métricas por Categoría - {metricas["nombre"]}\n'
                     f'Accuracy General: {metricas["test_accuracy"]:.4f}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.categorias, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if guardar:
            filename = f'metricas_categoria_{metricas["nombre"].lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico guardado: '{filename}'")
        
        plt.close()

    def comparar_modelos_tabla(self):
        """Genera tabla comparativa de todos los modelos"""
        print("\n" + "="*80)
        print("📊 TABLA COMPARATIVA DE MODELOS")
        print("="*80)

        datos_tabla = []
        for nombre, metricas in self.metricas_todas.items():
            datos_tabla.append({
                'Modelo': nombre,
                'Train Acc': f"{metricas['train_accuracy']:.4f}",
                'Test Acc': f"{metricas['test_accuracy']:.4f}",
                'Precision': f"{metricas['precision']:.4f}",
                'Recall': f"{metricas['recall']:.4f}",
                'F1-Score': f"{metricas['f1_score']:.4f}",
                'Tiempo (s)': f"{metricas['tiempo_entrenamiento']:.2f}"
            })

        df_comparacion = pd.DataFrame(datos_tabla)
        print("\n" + df_comparacion.to_string(index=False))

        df_comparacion.to_csv('comparacion_modelos.csv', index=False)
        print("\n💾 Tabla guardada en: comparacion_modelos.csv")

    def visualizar_comparacion_modelos(self, guardar=True):
        """Gráfico comparativo entre Random Forest y Logistic Regression"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        modelos = list(self.metricas_todas.keys())
        
        # Gráfico 1: Comparación de Accuracy
        train_accs = [m['train_accuracy'] for m in self.metricas_todas.values()]
        test_accs = [m['test_accuracy'] for m in self.metricas_todas.values()]
        
        x = np.arange(len(modelos))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, train_accs, width, label='Train', color='steelblue')
        bars2 = ax1.bar(x + width/2, test_accs, width, label='Test', color='darkorange')
        
        ax1.set_xlabel('Modelo', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Comparación de Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modelos, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=10)
        
        # Gráfico 2: Métricas promedio
        metricas_nombres = ['Precision', 'Recall', 'F1-Score']
        
        for i, (nombre, metricas) in enumerate(self.metricas_todas.items()):
            valores = [metricas['precision'], metricas['recall'], metricas['f1_score']]
            x2 = np.arange(len(metricas_nombres))
            offset = (i - len(self.metricas_todas)/2 + 0.5) * width
            
            color = 'steelblue' if i == 0 else 'darkorange'
            bars = ax2.bar(x2 + offset, valores, width, label=nombre, color=color)
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Métrica', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Comparación de Métricas Promedio', fontsize=14, fontweight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(metricas_nombres)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if guardar:
            plt.savefig('comparacion_modelos.png', dpi=300, bbox_inches='tight')
            print(f"\n✅ Comparación guardada: 'comparacion_modelos.png'")
        
        plt.close()

    # ------------------------------------------------------------------
    # ANÁLISIS TF-IDF
    # ------------------------------------------------------------------
    def palabras_importantes_por_categoria(self, top_n=15):
        """Muestra las palabras más importantes de cada categoría"""
        print(f"\n{'='*80}")
        print(f"⭐ TOP {top_n} PALABRAS MÁS IMPORTANTES POR CATEGORÍA (TF-IDF)")
        print(f"{'='*80}")

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Crear matriz TF-IDF completa para análisis
        X_full = self.vectorizer.transform(self.df['texto_limpio'])
        y_full = self.df['categoria'].values

        for categoria in sorted(np.unique(y_full)):
            print(f"\n🔹 {categoria.upper()}:")

            # Filtrar artículos de esta categoría
            indices = np.where(y_full == categoria)[0]

            # Sumar TF-IDF de todos los documentos de esta categoría
            categoria_tfidf = X_full[indices].sum(axis=0).A1

            # Obtener las top N palabras
            top_indices = categoria_tfidf.argsort()[-top_n:][::-1]
            top_palabras = feature_names[top_indices]
            top_scores = categoria_tfidf[top_indices]

            for i, (palabra, score) in enumerate(zip(top_palabras, top_scores), 1):
                print(f"   {i:2d}. {palabra:30s} (score: {score:.2f})")

    def visualizar_palabras_importantes(self, top_n=10, guardar=True):
        """Genera gráfico de barras con las palabras más importantes por categoría"""
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        X_full = self.vectorizer.transform(self.df['texto_limpio'])
        y_full = self.df['categoria'].values
        
        categorias_unicas = sorted(np.unique(y_full))
        n_cats = len(categorias_unicas)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, categoria in enumerate(categorias_unicas):
            indices = np.where(y_full == categoria)[0]
            categoria_tfidf = X_full[indices].sum(axis=0).A1
            
            top_indices = categoria_tfidf.argsort()[-top_n:][::-1]
            top_palabras = feature_names[top_indices]
            top_scores = categoria_tfidf[top_indices]
            
            # Crear gráfico de barras horizontal
            y_pos = np.arange(len(top_palabras))
            axes[idx].barh(y_pos, top_scores, color='steelblue', edgecolor='black')
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(top_palabras)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Score TF-IDF', fontweight='bold')
            axes[idx].set_title(f'Top {top_n} Palabras - {categoria}', 
                               fontsize=12, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if guardar:
            plt.savefig('palabras_importantes_tfidf.png', dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico guardado: 'palabras_importantes_tfidf.png'")
        
        plt.close()

    # ------------------------------------------------------------------
    # PREDICCIONES DE EJEMPLO
    # ------------------------------------------------------------------
    def predecir_ejemplos(self, modelo, nombre_modelo, num_ejemplos=5):
        """Muestra predicciones de ejemplo del conjunto de test"""
        print(f"\n{'='*80}")
        print(f"🔮 PREDICCIONES DE EJEMPLO - {nombre_modelo.upper()}")
        print(f"{'='*80}")
        
        # Tomar ejemplos aleatorios del conjunto de test
        indices_aleatorios = np.random.choice(len(self.y_test), num_ejemplos, replace=False)
        
        for i, idx in enumerate(indices_aleatorios, 1):
            # Obtener índice original del DataFrame
            idx_original = self.indices_test[idx]
            
            # Buscar el texto original usando el índice correcto
            fila = self.df.loc[idx_original]
            texto_original = fila.get('texto_completo', fila.get('titulo', fila.get('texto_limpio', '')))
            
            categoria_real = self.y_test[idx]
            categoria_predicha = modelo.predict(self.X_test[idx])[0]
            
            # Probabilidades
            probabilidades = modelo.predict_proba(self.X_test[idx])[0]
            
            print(f"\n{'─'*80}")
            print(f"EJEMPLO {i}:")
            print(f"{'─'*80}")
            print(f"📰 Texto: {texto_original[:250]}...")
            print(f"\n✅ Categoría Real:     {categoria_real}")
            print(f"🤖 Categoría Predicha: {categoria_predicha}")
            
            correcto = "✅ CORRECTO" if categoria_real == categoria_predicha else "❌ INCORRECTO"
            print(f"   {correcto}")
            
            print(f"\n📊 Probabilidades por categoría:")
            for cat, prob in zip(self.categorias, probabilidades):
                barra = '█' * int(prob * 40)
                espacios = ' ' * (40 - len(barra))
                porcentaje = prob * 100
                print(f"   {cat:<20} {prob:.4f} ({porcentaje:5.1f}%) |{barra}{espacios}|")

    def predecir_texto_custom(self, modelo, nombre_modelo, texto):
        """Predice la categoría de un texto personalizado"""
        print(f"\n{'='*80}")
        print(f"🔮 PREDICCIÓN DE TEXTO PERSONALIZADO - {nombre_modelo.upper()}")
        print(f"{'='*80}")
        
        # Vectorizar el texto
        texto_vectorizado = self.vectorizer.transform([texto])
        
        # Predecir
        categoria_predicha = modelo.predict(texto_vectorizado)[0]
        probabilidades = modelo.predict_proba(texto_vectorizado)[0]
        
        print(f"\n📰 Texto ingresado:")
        print(f"   {texto}")
        
        print(f"\n🤖 Categoría Predicha: {categoria_predicha}")
        
        print(f"\n📊 Probabilidades por categoría:")
        for cat, prob in zip(self.categorias, probabilidades):
            barra = '█' * int(prob * 40)
            espacios = ' ' * (40 - len(barra))
            porcentaje = prob * 100
            print(f"   {cat:<20} {prob:.4f} ({porcentaje:5.1f}%) |{barra}{espacios}|")

    # ------------------------------------------------------------------
    # GUARDAR MODELO
    # ------------------------------------------------------------------
    def guardar_modelo(self, modelo, nombre):
        """Guarda el modelo junto con el vectorizer y categorías"""
        paquete = {
            'modelo': modelo,
            'vectorizer': self.vectorizer,
            'categorias': self.categorias
        }
        joblib.dump(paquete, nombre)
        print(f"💾 Modelo guardado en: {nombre}")

    # ------------------------------------------------------------------
    # PIPELINE COMPLETO
    # ------------------------------------------------------------------
    def ejecutar_pipeline_completo(self, archivo_csv='abc_news_limpio.csv'):
        """Ejecuta el pipeline completo de clasificación"""
        print("\n" + "🚀"*40)
        print("PIPELINE COMPLETO DE CLASIFICACIÓN - RANDOM FOREST vs LOGISTIC REGRESSION")
        print("🚀"*40 + "\n")

        # 1. Cargar datos preprocesados
        self.cargar_datos_preprocesados(archivo_csv)
        self.aplicar_tfidf_y_dividir()

        # 2. Análisis TF-IDF - Palabras más importantes
        print("\n" + "⭐"*40)
        print("ANÁLISIS TF-IDF")
        print("⭐"*40)
        self.palabras_importantes_por_categoria(top_n=15)
        self.visualizar_palabras_importantes(top_n=10)

        # 3. Entrenar ambos modelos
        self.entrenar_random_forest()
        self.entrenar_logistic_regression()

        # 4. Evaluar modelos
        print("\n" + "📊"*40)
        print("EVALUACIÓN DE MODELOS")
        print("📊"*40)
        
        metricas_rf = self.evaluar_modelo(self.modelo_rf, "Random Forest")
        metricas_lr = self.evaluar_modelo(self.modelo_lr, "Logistic Regression")

        # 5. Validación cruzada
        print("\n" + "🔄"*40)
        print("VALIDACIÓN CRUZADA")
        print("🔄"*40)
        
        self.validacion_cruzada(self.modelo_rf, "Random Forest")
        self.validacion_cruzada(self.modelo_lr, "Logistic Regression")

        # 6. Predicciones de ejemplo
        print("\n" + "🔮"*40)
        print("PREDICCIONES DE EJEMPLO")
        print("🔮"*40)
        
        self.predecir_ejemplos(self.modelo_rf, "Random Forest", num_ejemplos=5)
        self.predecir_ejemplos(self.modelo_lr, "Logistic Regression", num_ejemplos=5)
        
        # Ejemplos de predicción con textos personalizados
        print("\n" + "="*80)
        print("🧪 PREDICCIONES CON TEXTOS PERSONALIZADOS")
        print("="*80)
        
        textos_prueba = [
            "El presidente estadosunidos anunció nuevas medidas económicas para combatir la inflación",
            "La inteligencia artificial revoluciona el sector tecnológico con nuevos avances en aprendizaje automático",
            "El museo presenta una nueva exposición de arte contemporáneo con obras de artistas emergentes",
            "Las bolsas europeas cierran con ganancias tras los datos positivos del PIB"
        ]
        
        mejor_modelo = self.modelo_rf if metricas_rf['test_accuracy'] > metricas_lr['test_accuracy'] else self.modelo_lr
        mejor_nombre = "Random Forest" if metricas_rf['test_accuracy'] > metricas_lr['test_accuracy'] else "Logistic Regression"
        
        for texto in textos_prueba:
            self.predecir_texto_custom(mejor_modelo, mejor_nombre, texto)

        # 7. Generar visualizaciones
        print("\n" + "="*80)
        print("📊 GENERANDO VISUALIZACIONES")
        print("="*80)
        
        self.visualizar_matriz_confusion(metricas_rf)
        self.visualizar_matriz_confusion(metricas_lr)
        self.visualizar_metricas_por_categoria(metricas_rf)
        self.visualizar_metricas_por_categoria(metricas_lr)
        self.visualizar_comparacion_modelos()

        # 8. Tabla comparativa
        self.comparar_modelos_tabla()

        # 9. Guardar modelos
        print("\n" + "="*80)
        print("💾 GUARDANDO MODELOS")
        print("="*80)
        self.guardar_modelo(self.modelo_rf, "modelo_randomforest.pkl")
        self.guardar_modelo(self.modelo_lr, "modelo_logistic_regression.pkl")

        # 10. Resumen final
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*80)
        
        print(f"\n📁 Archivos generados:")
        print(f"   - palabras_importantes_tfidf.png")
        print(f"   - matriz_confusion_random_forest.png")
        print(f"   - matriz_confusion_logistic_regression.png")
        print(f"   - metricas_categoria_random_forest.png")
        print(f"   - metricas_categoria_logistic_regression.png")
        print(f"   - comparacion_modelos.png")
        print(f"   - comparacion_modelos.csv")
        print(f"   - modelo_randomforest.pkl")
        print(f"   - modelo_logistic_regression.pkl")

        # Mejor modelo
        mejor_nombre = max(self.metricas_todas.items(), 
                          key=lambda x: x[1]['test_accuracy'])[0]
        mejor_acc = self.metricas_todas[mejor_nombre]['test_accuracy']

        print(f"\n🏆 MEJOR MODELO: {mejor_nombre}")
        print(f"   Test Accuracy: {mejor_acc:.4f} ({mejor_acc*100:.2f}%)")
        
        # Comparación de diferencia
        if len(self.metricas_todas) == 2:
            accs = [m['test_accuracy'] for m in self.metricas_todas.values()]
            diferencia = abs(accs[0] - accs[1]) * 100
            print(f"\n📊 Diferencia entre modelos: {diferencia:.2f}%")

        print("\n" + "🎉"*40)
        print("¡CLASIFICADOR DE NOTICIAS COMPLETADO!")
        print("🎉"*40 + "\n")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    clasificador = ClasificadorNoticias()
    
    # Ejecutar pipeline con el CSV generado por preprocesamiento.py
    clasificador.ejecutar_pipeline_completo('abc_news_limpio.csv')