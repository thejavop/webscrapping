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
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import warnings
import time

warnings.filterwarnings('ignore')

# Importar la clase de preprocesamiento
from preprocesamiento import PreprocesadorNoticias


class ClasificadorNoticias:
    """
    Clasificador de noticias usando TF-IDF + RandomForestClassifier + LogisticRegression
    (Ejercicio 3 - RA4)
    
    Compara dos modelos:
    - Random Forest (árbol de decisiones múltiple)
    - Logistic Regression (modelo lineal)
    """

    def __init__(self):  # ✅ CORREGIDO: doble guión bajo
        """Inicializa el clasificador"""
        self.modelo = None
        self.mejor_modelo = None
        self.modelo_logistic = None
        self.preprocessor = None
        self.categorias = None
        self.tiempo_entrenamiento = {}  # Diccionario para almacenar tiempos
        self.metricas_todas = {}  # Para almacenar todas las métricas

    # ------------------------------------------------------------------
    # CARGA Y PREPROCESAMIENTO
    # ------------------------------------------------------------------
    def cargar_datos_preprocesados(self):
        """Carga el CSV y ejecuta el pipeline de preprocesamiento"""
        print("="*80)
        print("🔄 CARGANDO Y PREPROCESANDO DATOS")
        print("="*80)

        self.preprocessor = PreprocesadorNoticias('abc_news.csv')
        self.preprocessor.ejecutar_pipeline_completo()

        self.X_train = self.preprocessor.X_train
        self.X_test = self.preprocessor.X_test
        self.y_train = self.preprocessor.y_train
        self.y_test = self.preprocessor.y_test

        self.categorias = sorted(np.unique(self.y_train))

        print("\n✅ Datos cargados correctamente")
        print(f"   - Categorías: {self.categorias}")

    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------
    def entrenar_modelo_base(self):
        """Entrena Random Forest con parámetros por defecto"""
        print("\n" + "="*80)
        print("🌲 ENTRENANDO RANDOM FOREST (MODELO BASE)")
        print("="*80)
        print("Parámetros: n_estimators=100, random_state=42")

        self.modelo = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,  # Usa todos los cores del CPU
            verbose=0
        )

        print("\n⏳ Entrenando...")
        inicio = time.time()
        self.modelo.fit(self.X_train, self.y_train)
        tiempo = time.time() - inicio
        self.tiempo_entrenamiento['rf_base'] = tiempo

        print(f"✅ Random Forest entrenado en {tiempo:.2f} segundos")

    def entrenar_logistic_regression(self):
        """Entrena Logistic Regression para comparar"""
        print("\n" + "="*80)
        print("📊 ENTRENANDO LOGISTIC REGRESSION (COMPARACIÓN)")
        print("="*80)
        print("Parámetros: max_iter=1000, solver='lbfgs'")

        self.modelo_logistic = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            verbose=0
        )

        print("\n⏳ Entrenando...")
        inicio = time.time()
        self.modelo_logistic.fit(self.X_train, self.y_train)
        tiempo = time.time() - inicio
        self.tiempo_entrenamiento['logistic'] = tiempo

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

        # Tiempo de entrenamiento
        tiempo_key = None
        if 'Random Forest Base' in nombre:
            tiempo_key = 'rf_base'
        elif 'Logistic' in nombre:
            tiempo_key = 'logistic'
        elif 'Optimizado' in nombre:
            tiempo_key = 'rf_optimizado'

        tiempo = self.tiempo_entrenamiento.get(tiempo_key, 0)

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

        # Guardar en diccionario global
        self.metricas_todas[nombre] = metricas

        return metricas

    # ------------------------------------------------------------------
    # OPTIMIZACIÓN
    # ------------------------------------------------------------------
    def optimizar_hiperparametros_randomforest(self):
        """Optimiza hiperparámetros de Random Forest con GridSearchCV"""
        print("\n" + "="*80)
        print("🔧 OPTIMIZANDO RANDOM FOREST CON GRIDSEARCHCV")
        print("="*80)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 30],
            'min_samples_split': [2, 5]
        }

        print("\n🔍 Parámetros a probar:")
        for param, values in param_grid.items():
            print(f"   - {param}: {values}")

        total_combinaciones = np.prod([len(v) for v in param_grid.values()])
        print(f"\n📊 Total de combinaciones: {total_combinaciones}")
        print(f"⏳ Esto puede tardar varios minutos...")

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,  # 3-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        inicio = time.time()
        grid.fit(self.X_train, self.y_train)
        tiempo = time.time() - inicio
        self.tiempo_entrenamiento['rf_optimizado'] = tiempo

        self.mejor_modelo = grid.best_estimator_

        print(f"\n✅ Optimización completada en {tiempo:.2f} segundos")
        print(f"\n🏆 MEJORES HIPERPARÁMETROS:")
        for param, value in grid.best_params_.items():
            print(f"   - {param}: {value}")
        print(f"\n📊 Mejor Score (CV): {grid.best_score_:.4f}")

        return self.mejor_modelo

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

    def comparar_modelos_tabla(self):
        """Genera tabla comparativa de todos los modelos"""
        print("\n" + "="*80)
        print("📊 TABLA COMPARATIVA DE MODELOS")
        print("="*80)

        # Crear DataFrame con las métricas
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

        # Guardar como CSV
        df_comparacion.to_csv('comparacion_modelos.csv', index=False)
        print("\n💾 Tabla guardada en: comparacion_modelos.csv")

    def visualizar_comparacion_accuracy(self, guardar=True):
        """Gráfico de barras comparando accuracy de los modelos"""
        modelos = list(self.metricas_todas.keys())
        test_accs = [m['test_accuracy'] for m in self.metricas_todas.values()]

        plt.figure(figsize=(12, 6))
        
        colores = ['steelblue', 'darkorange', 'green']
        bars = plt.bar(modelos, test_accs, color=colores[:len(modelos)], edgecolor='black')

        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

        plt.xlabel('Modelo', fontsize=12, fontweight='bold')
        plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        plt.title('Comparación de Accuracy entre Modelos', fontsize=14, fontweight='bold', pad=20)
        plt.ylim([0, 1.1])
        plt.xticks(rotation=15, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if guardar:
            plt.savefig('comparacion_accuracy.png', dpi=300, bbox_inches='tight')
            print("   ✅ Guardado: comparacion_accuracy.png")

        plt.close()

    # ------------------------------------------------------------------
    # GUARDAR MODELO
    # ------------------------------------------------------------------
    def guardar_modelo(self, modelo, nombre):
        """Guarda el modelo junto con el vectorizer y categorías"""
        paquete = {
            'modelo': modelo,
            'vectorizer': self.preprocessor.vectorizer,
            'categorias': self.categorias
        }
        joblib.dump(paquete, nombre)
        print(f"💾 Modelo guardado en: {nombre}")

    # ------------------------------------------------------------------
    # PIPELINE COMPLETO
    # ------------------------------------------------------------------
    def ejecutar_pipeline_completo(self):
        """Ejecuta el pipeline completo de clasificación"""
        print("\n" + "🚀"*40)
        print("PIPELINE COMPLETO DE CLASIFICACIÓN - RANDOM FOREST vs LOGISTIC REGRESSION")
        print("🚀"*40 + "\n")

        # 1. Cargar datos
        self.cargar_datos_preprocesados()

        # 2. Entrenar modelos base
        self.entrenar_modelo_base()
        self.entrenar_logistic_regression()

        # 3. Evaluar modelos base
        print("\n" + "📊"*40)
        print("EVALUACIÓN DE MODELOS BASE")
        print("📊"*40)
        
        metricas_rf = self.evaluar_modelo(self.modelo, "Random Forest Base")
        metricas_lr = self.evaluar_modelo(self.modelo_logistic, "Logistic Regression")

        # 4. Optimizar Random Forest
        mejor_rf = self.optimizar_hiperparametros_randomforest()
        metricas_rf_opt = self.evaluar_modelo(mejor_rf, "Random Forest Optimizado")

        # 5. Generar visualizaciones
        print("\n" + "="*80)
        print("📊 GENERANDO VISUALIZACIONES")
        print("="*80)
        
        self.visualizar_matriz_confusion(metricas_rf)
        self.visualizar_matriz_confusion(metricas_lr)
        self.visualizar_matriz_confusion(metricas_rf_opt)
        self.visualizar_comparacion_accuracy()

        # 6. Tabla comparativa
        self.comparar_modelos_tabla()

        # 7. Guardar modelos
        print("\n" + "="*80)
        print("💾 GUARDANDO MODELOS")
        print("="*80)
        self.guardar_modelo(mejor_rf, "modelo_randomforest_optimizado.pkl")
        self.guardar_modelo(self.modelo_logistic, "modelo_logistic_regression.pkl")

        # 8. Resumen final
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*80)
        
        print(f"\n📁 Archivos generados:")
        print(f"   - matriz_confusion_random_forest_base.png")
        print(f"   - matriz_confusion_logistic_regression.png")
        print(f"   - matriz_confusion_random_forest_optimizado.png")
        print(f"   - comparacion_accuracy.png")
        print(f"   - comparacion_modelos.csv")
        print(f"   - modelo_randomforest_optimizado.pkl")
        print(f"   - modelo_logistic_regression.pkl")

        # Mejor modelo
        mejor_nombre = max(self.metricas_todas.items(), 
                          key=lambda x: x[1]['test_accuracy'])[0]
        mejor_acc = self.metricas_todas[mejor_nombre]['test_accuracy']

        print(f"\n🏆 MEJOR MODELO: {mejor_nombre}")
        print(f"   Test Accuracy: {mejor_acc:.4f} ({mejor_acc*100:.2f}%)")

        print("\n" + "🎉"*40)
        print("¡CLASIFICADOR DE NOTICIAS COMPLETADO!")
        print("🎉"*40 + "\n")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    clasificador = ClasificadorNoticias()
    clasificador.ejecutar_pipeline_completo()