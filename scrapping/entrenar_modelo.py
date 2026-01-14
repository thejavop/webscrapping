import matplotlib
matplotlib.use('Agg')

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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings
import time
import os

warnings.filterwarnings('ignore')


class EntrenadorModelo:
    """
    Clase para entrenar modelos de clasificación de noticias.

    Entrena Random Forest y Logistic Regression, genera visualizaciones
    y guarda los modelos entrenados para su uso posterior.
    """

    def __init__(self):
        self.modelo_rf = None
        self.modelo_lr = None
        self.vectorizer = None
        self.categorias = None
        self.tiempo_entrenamiento = {}
        self.metricas_todas = {}

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.indices_train = None
        self.indices_test = None
        self.df = None
        self.columna_texto = None

    def cargar_datos_preprocesados(self, archivo_csv='abc_news_limpio.csv'):
        """Carga el CSV ya preprocesado"""
        print("="*80)
        print("CARGANDO DATOS PREPROCESADOS")
        print("="*80)

        if not os.path.exists(archivo_csv):
            raise FileNotFoundError(
                f"No se encuentra '{archivo_csv}'.\n"
                f"Debes ejecutar primero 'preprocesamiento.py' para generar el CSV limpio."
            )
        self.df = pd.read_csv(archivo_csv)

        print(f"CSV cargado: {len(self.df)} articulos")
        print(f"Categorias encontradas: {self.df['categoria'].unique().tolist()}")

        posibles_columnas_texto = ['texto_limpio', 'descripcion', 'texto_completo', 'titulo']
        self.columna_texto = next((col for col in posibles_columnas_texto if col in self.df.columns), None)
        if not self.columna_texto:
            raise ValueError(f"No se encontro ninguna columna de texto valida en el CSV.")

        print(f"Columna de texto usada para TF-IDF: {self.columna_texto}")
        print(f"\nDistribucion por categoria:")
        print(self.df['categoria'].value_counts())

        return self.df

    def aplicar_tfidf_y_dividir(self, test_size=0.2, random_state=42):
        """Aplica TF-IDF y divide en train/test"""
        print("\n" + "="*80)
        print("APLICANDO TF-IDF Y DIVIDIENDO DATOS")
        print("="*80)

        X_texto = self.df[self.columna_texto]
        y = self.df['categoria'].values
        indices = self.df.index.values

        print("\nConfigurando TF-IDF Vectorizer:")
        print("   - max_features: 1500")
        print("   - ngram_range: (1, 2)")
        print("   - min_df: 2")

        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            min_df=2
        )

        print("\nTransformando textos a vectores TF-IDF...")
        X = self.vectorizer.fit_transform(X_texto)

        print(f"TF-IDF aplicado - Matriz shape: {X.shape}")

        print(f"\nDividiendo datos ({int((1-test_size)*100)}% train / {int(test_size*100)}% test)...")

        self.X_train, self.X_test, self.y_train, self.y_test, self.indices_train, self.indices_test = train_test_split(
            X, y, indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        self.categorias = sorted(np.unique(self.y_train))

        print(f"Division completada:")
        print(f"   - Train: {self.X_train.shape[0]} articulos")
        print(f"   - Test:  {self.X_test.shape[0]} articulos")

    def entrenar_random_forest(self):
        """Entrena Random Forest"""
        print("\n" + "="*80)
        print("ENTRENANDO RANDOM FOREST")
        print("="*80)

        self.modelo_rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        inicio = time.time()
        self.modelo_rf.fit(self.X_train, self.y_train)
        tiempo = time.time() - inicio
        self.tiempo_entrenamiento['Random Forest'] = tiempo

        print(f"Random Forest entrenado en {tiempo:.2f} segundos")

    def entrenar_logistic_regression(self):
        """Entrena Logistic Regression"""
        print("\n" + "="*80)
        print("ENTRENANDO LOGISTIC REGRESSION")
        print("="*80)

        self.modelo_lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )

        inicio = time.time()
        self.modelo_lr.fit(self.X_train, self.y_train)
        tiempo = time.time() - inicio
        self.tiempo_entrenamiento['Logistic Regression'] = tiempo

        print(f"Logistic Regression entrenado en {tiempo:.2f} segundos")

    def evaluar_modelo(self, modelo, nombre):
        """Evalua el modelo y retorna metricas"""
        print("\n" + "="*80)
        print(f"EVALUANDO {nombre.upper()}")
        print("="*80)

        y_pred_train = modelo.predict(self.X_train)
        y_pred_test = modelo.predict(self.X_test)

        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)

        print(f"\nACCURACY:")
        print(f"   - Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   - Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")

        print(f"\nREPORTE DE CLASIFICACION (Test):")
        print("-"*80)
        print(classification_report(
            self.y_test,
            y_pred_test,
            target_names=self.categorias,
            digits=4
        ))

        cm = confusion_matrix(self.y_test, y_pred_test, labels=self.categorias)

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred_test, average='weighted'
        )

        tiempo = self.tiempo_entrenamiento.get(nombre, 0)

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
        """Realiza validacion cruzada de 5-fold"""
        print(f"\n{'='*80}")
        print(f"VALIDACION CRUZADA (5-FOLD) - {nombre}")
        print(f"{'='*80}")

        scores = cross_val_score(
            modelo, self.X_train, self.y_train,
            cv=5, scoring='accuracy', n_jobs=-1
        )

        print(f"\nScores por Fold:")
        for i, score in enumerate(scores, 1):
            print(f"   Fold {i}: {score:.4f} ({score*100:.2f}%)")

        print(f"\nRESUMEN:")
        print(f"   - Media: {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
        print(f"   - Desviacion estandar: {scores.std():.4f}")

        return scores

    def visualizar_matriz_confusion(self, metricas, guardar=True):
        """Genera heatmap de la matriz de confusion"""
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            metricas['confusion_matrix'],
            annot=True, fmt='d', cmap='Blues',
            xticklabels=self.categorias,
            yticklabels=self.categorias
        )

        plt.title(f"Matriz de Confusion - {metricas['nombre']}\nAccuracy: {metricas['test_accuracy']:.4f}")
        plt.ylabel('Categoria Real')
        plt.xlabel('Categoria Predicha')
        plt.tight_layout()

        if guardar:
            filename = f"matriz_confusion_{metricas['nombre'].lower().replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Guardado: {filename}")

        plt.close()

    def visualizar_comparacion_modelos(self, guardar=True):
        """Grafico comparativo entre modelos"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        modelos = list(self.metricas_todas.keys())
        train_accs = [m['train_accuracy'] for m in self.metricas_todas.values()]
        test_accs = [m['test_accuracy'] for m in self.metricas_todas.values()]

        x = np.arange(len(modelos))
        width = 0.35

        ax1.bar(x - width/2, train_accs, width, label='Train', color='steelblue')
        ax1.bar(x + width/2, test_accs, width, label='Test', color='darkorange')
        ax1.set_xlabel('Modelo')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Comparacion de Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modelos)
        ax1.legend()
        ax1.set_ylim([0, 1.1])

        metricas_nombres = ['Precision', 'Recall', 'F1-Score']
        for i, (nombre, metricas) in enumerate(self.metricas_todas.items()):
            valores = [metricas['precision'], metricas['recall'], metricas['f1_score']]
            x2 = np.arange(len(metricas_nombres))
            offset = (i - len(self.metricas_todas)/2 + 0.5) * width
            color = 'steelblue' if i == 0 else 'darkorange'
            ax2.bar(x2 + offset, valores, width, label=nombre, color=color)

        ax2.set_xlabel('Metrica')
        ax2.set_ylabel('Score')
        ax2.set_title('Comparacion de Metricas')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(metricas_nombres)
        ax2.legend()
        ax2.set_ylim([0, 1.1])

        plt.tight_layout()

        if guardar:
            plt.savefig('comparacion_modelos.png', dpi=300, bbox_inches='tight')
            print(f"Guardado: comparacion_modelos.png")

        plt.close()

    def guardar_modelo(self, modelo, nombre_archivo):
        """Guarda el modelo junto con el vectorizer y categorias"""
        paquete = {
            'modelo': modelo,
            'vectorizer': self.vectorizer,
            'categorias': self.categorias
        }
        joblib.dump(paquete, nombre_archivo)
        print(f"Modelo guardado en: {nombre_archivo}")

    def ejecutar_pipeline_completo(self, archivo_csv='abc_news_limpio.csv'):
        """Ejecuta el pipeline completo de entrenamiento"""
        print("\n" + "="*80)
        print("PIPELINE DE ENTRENAMIENTO - RANDOM FOREST vs LOGISTIC REGRESSION")
        print("="*80 + "\n")

        # 1. Cargar datos
        self.cargar_datos_preprocesados(archivo_csv)
        self.aplicar_tfidf_y_dividir()

        # 2. Entrenar modelos
        self.entrenar_random_forest()
        self.entrenar_logistic_regression()

        # 3. Evaluar modelos
        metricas_rf = self.evaluar_modelo(self.modelo_rf, "Random Forest")
        metricas_lr = self.evaluar_modelo(self.modelo_lr, "Logistic Regression")

        # 4. Validacion cruzada
        self.validacion_cruzada(self.modelo_rf, "Random Forest")
        self.validacion_cruzada(self.modelo_lr, "Logistic Regression")

        # 5. Visualizaciones
        print("\n" + "="*80)
        print("GENERANDO VISUALIZACIONES")
        print("="*80)
        self.visualizar_matriz_confusion(metricas_rf)
        self.visualizar_matriz_confusion(metricas_lr)
        self.visualizar_comparacion_modelos()

        # 6. Guardar modelos
        print("\n" + "="*80)
        print("GUARDANDO MODELOS")
        print("="*80)
        self.guardar_modelo(self.modelo_rf, "modelo_randomforest.pkl")
        self.guardar_modelo(self.modelo_lr, "modelo_logistic_regression.pkl")

        # 7. Resumen final
        mejor_nombre = max(self.metricas_todas.items(),
                          key=lambda x: x[1]['test_accuracy'])[0]
        mejor_acc = self.metricas_todas[mejor_nombre]['test_accuracy']

        print("\n" + "="*80)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*80)
        print(f"\nMEJOR MODELO: {mejor_nombre}")
        print(f"Test Accuracy: {mejor_acc:.4f} ({mejor_acc*100:.2f}%)")
        print(f"\nArchivos generados:")
        print(f"   - modelo_randomforest.pkl")
        print(f"   - modelo_logistic_regression.pkl")
        print(f"   - matriz_confusion_random_forest.png")
        print(f"   - matriz_confusion_logistic_regression.png")
        print(f"   - comparacion_modelos.png")


if __name__ == "__main__":
    entrenador = EntrenadorModelo()
    entrenador.ejecutar_pipeline_completo('abc_news_limpio.csv')
