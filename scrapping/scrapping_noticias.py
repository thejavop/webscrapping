import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
warnings.filterwarnings('ignore')

# Importar la clase de preprocesamiento
from preprocesamiento import PreprocesadorNoticias


class ClasificadorNoticias:
    """
    Clasificador de noticias usando TF-IDF + Logistic Regression
    
    Este modelo clasifica noticias de ABC.es en 4 categorías:
    - Internacional
    - Economía
    - Tecnología
    - Cultura
    """
    
    def __init__(self):
        self.modelo = None
        self.mejor_modelo = None
        self.preprocessor = None
        self.categorias = None
        self.metricas = {}
        
    def cargar_datos_preprocesados(self):
        """Carga y preprocesa los datos usando la clase PreprocesadorNoticias"""
        print("="*80)
        print("🔄 CARGANDO Y PREPROCESANDO DATOS")
        print("="*80)
        
        # Ejecutar pipeline de preprocesamiento
        self.preprocessor = PreprocesadorNoticias('abc_news.csv')
        self.preprocessor.ejecutar_pipeline_completo()
        
        # Obtener datos procesados
        self.X_train = self.preprocessor.X_train
        self.X_test = self.preprocessor.X_test
        self.y_train = self.preprocessor.y_train
        self.y_test = self.preprocessor.y_test
        
        # Guardar categorías únicas
        self.categorias = sorted(np.unique(self.y_train))
        
        print("\n✅ Datos cargados correctamente")
        
    def entrenar_modelo_base(self):
        """Entrena un modelo LogisticRegression básico"""
        print("\n" + "="*80)
        print("🤖 ENTRENANDO MODELO BASE: Logistic Regression")
        print("="*80)
        
        # Crear modelo con parámetros por defecto
        self.modelo = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',  # Para múltiples categorías
            solver='lbfgs',
            verbose=0
        )
        
        print("\n⏳ Entrenando modelo...")
        self.modelo.fit(self.X_train, self.y_train)
        print("✅ Modelo entrenado exitosamente")
        
    def evaluar_modelo(self, modelo, nombre_modelo="Modelo"):
        """Evalúa el modelo y calcula todas las métricas"""
        print(f"\n{'='*80}")
        print(f"📊 EVALUANDO {nombre_modelo.upper()}")
        print(f"{'='*80}")
        
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
        report = classification_report(
            self.y_test, 
            y_pred_test, 
            target_names=self.categorias,
            digits=4
        )
        print(report)
        
        # Métricas por categoría
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, 
            y_pred_test, 
            labels=self.categorias,
            average=None
        )
        
        print(f"\n📈 MÉTRICAS DETALLADAS POR CATEGORÍA:")
        print("-"*80)
        print(f"{'Categoría':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Soporte':<10}")
        print("-"*80)
        for i, cat in enumerate(self.categorias):
            print(f"{cat:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
        
        # Promedios
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            self.y_test, 
            y_pred_test,
            average='weighted'
        )
        
        print("-"*80)
        print(f"{'PROMEDIO (weighted)':<20} {precision_avg:<12.4f} {recall_avg:<12.4f} {f1_avg:<12.4f}")
        print("-"*80)
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred_test, labels=self.categorias)
        
        # Guardar métricas
        metricas = {
            'nombre': nombre_modelo,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision_avg,
            'recall': recall_avg,
            'f1_score': f1_avg,
            'confusion_matrix': cm,
            'y_pred_test': y_pred_test
        }
        
        return metricas
    
    def optimizar_hiperparametros(self):
        """Optimiza hiperparámetros usando GridSearchCV"""
        print("\n" + "="*80)
        print("🔧 OPTIMIZANDO HIPERPARÁMETROS CON GRIDSEARCHCV")
        print("="*80)
        
        # Definir grilla de parámetros
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],  # Regularización
            'penalty': ['l2'],  # Tipo de regularización
            'solver': ['lbfgs', 'saga'],  # Algoritmo de optimización
            'max_iter': [1000, 2000]
        }
        
        print("\n🔍 Parámetros a probar:")
        for param, values in param_grid.items():
            print(f"   - {param}: {values}")
        
        total_combinaciones = np.prod([len(v) for v in param_grid.values()])
        print(f"\n📊 Total de combinaciones: {total_combinaciones}")
        print(f"⏳ Esto puede tardar varios minutos...")
        
        # GridSearchCV
        grid_search = GridSearchCV(
            LogisticRegression(
                random_state=42,
                multi_class='multinomial',
                verbose=0
            ),
            param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,  # Usar todos los cores
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Mejor modelo
        self.mejor_modelo = grid_search.best_estimator_
        
        print("\n✅ Optimización completada")
        print(f"\n🏆 MEJORES HIPERPARÁMETROS:")
        for param, value in grid_search.best_params_.items():
            print(f"   - {param}: {value}")
        
        print(f"\n📊 Mejor Score (CV): {grid_search.best_score_:.4f}")
        
        # Mostrar top 5 configuraciones
        print(f"\n🥇 TOP 5 CONFIGURACIONES:")
        print("-"*80)
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_5 = results_df.nlargest(5, 'mean_test_score')[
            ['params', 'mean_test_score', 'std_test_score']
        ]
        for idx, row in top_5.iterrows():
            print(f"{row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f}) - {row['params']}")
        
        return self.mejor_modelo
    
    def validacion_cruzada(self, modelo, nombre="Modelo"):
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
    
    def visualizar_matriz_confusion(self, metricas, guardar=True):
        """Genera visualización de la matriz de confusión"""
        plt.figure(figsize=(10, 8))
        
        cm = metricas['confusion_matrix']
        
        # Heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.categorias,
            yticklabels=self.categorias,
            cbar_kws={'label': 'Número de predicciones'}
        )
        
        plt.title(f'Matriz de Confusión - {metricas["nombre"]}\n'
                  f'Accuracy: {metricas["test_accuracy"]:.4f}',
                  fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Categoría Real', fontsize=12, fontweight='bold')
        plt.xlabel('Categoría Predicha', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if guardar:
            filename = f'matriz_confusion_{metricas["nombre"].lower().replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n✅ Matriz de confusión guardada: '{filename}'")
        
        plt.show()
    
    def visualizar_metricas_por_categoria(self, metricas, guardar=True):
        """Genera gráfico de barras con métricas por categoría"""
        # Calcular métricas por categoría
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test,
            metricas['y_pred_test'],
            labels=self.categorias,
            average=None
        )
        
        # Crear DataFrame para visualización
        df_metricas = pd.DataFrame({
            'Categoría': self.categorias,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Configurar gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.categorias))
        width = 0.25
        
        bars1 = ax.bar(x - width, df_metricas['Precision'], width, label='Precision', color='skyblue')
        bars2 = ax.bar(x, df_metricas['Recall'], width, label='Recall', color='lightcoral')
        bars3 = ax.bar(x + width, df_metricas['F1-Score'], width, label='F1-Score', color='lightgreen')
        
        # Añadir valores sobre las barras
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
            print(f"✅ Gráfico de métricas guardado: '{filename}'")
        
        plt.show()
    
    def comparar_modelos_visualizacion(self, metricas_base, metricas_optimizado, guardar=True):
        """Compara visualmente modelo base vs optimizado"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gráfico 1: Comparación de Accuracy
        modelos = ['Modelo Base', 'Modelo Optimizado']
        train_accs = [metricas_base['train_accuracy'], metricas_optimizado['train_accuracy']]
        test_accs = [metricas_base['test_accuracy'], metricas_optimizado['test_accuracy']]
        
        x = np.arange(len(modelos))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, train_accs, width, label='Train', color='steelblue')
        bars2 = ax1.bar(x + width/2, test_accs, width, label='Test', color='darkorange')
        
        ax1.set_xlabel('Modelo', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Comparación de Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modelos)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # Añadir valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=10)
        
        # Gráfico 2: Comparación de métricas promedio
        metricas_nombres = ['Precision', 'Recall', 'F1-Score']
        base_vals = [
            metricas_base['precision'],
            metricas_base['recall'],
            metricas_base['f1_score']
        ]
        opt_vals = [
            metricas_optimizado['precision'],
            metricas_optimizado['recall'],
            metricas_optimizado['f1_score']
        ]
        
        x2 = np.arange(len(metricas_nombres))
        bars3 = ax2.bar(x2 - width/2, base_vals, width, label='Base', color='steelblue')
        bars4 = ax2.bar(x2 + width/2, opt_vals, width, label='Optimizado', color='darkorange')
        
        ax2.set_xlabel('Métrica', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Comparación de Métricas Promedio', fontsize=14, fontweight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(metricas_nombres)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        # Añadir valores
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if guardar:
            plt.savefig('comparacion_modelos.png', dpi=300, bbox_inches='tight')
            print(f"\n✅ Comparación guardada: 'comparacion_modelos.png'")
        
        plt.show()
    
    def guardar_modelo(self, modelo, nombre_archivo='modelo_logistic_regression.pkl'):
        """Guarda el modelo entrenado"""
        joblib.dump(modelo, nombre_archivo)
        print(f"\n💾 Modelo guardado: '{nombre_archivo}'")
    
    def predecir_ejemplo(self, modelo, num_ejemplos=5):
        """Muestra predicciones de ejemplo"""
        print(f"\n{'='*80}")
        print(f"🔮 PREDICCIONES DE EJEMPLO")
        print(f"{'='*80}")
        
        # Tomar ejemplos aleatorios del conjunto de test
        indices_aleatorios = np.random.choice(len(self.y_test), num_ejemplos, replace=False)
        
        for i, idx in enumerate(indices_aleatorios, 1):
            # Obtener texto original
            texto_original = self.preprocessor.df.iloc[
                self.preprocessor.df.index[idx + len(self.y_train)]
            ]['texto_completo']
            
            categoria_real = self.y_test[idx]
            categoria_predicha = modelo.predict(self.X_test[idx])[0]
            
            # Probabilidades
            probabilidades = modelo.predict_proba(self.X_test[idx])[0]
            
            print(f"\n{'─'*80}")
            print(f"EJEMPLO {i}:")
            print(f"{'─'*80}")
            print(f"📰 Texto: {texto_original[:200]}...")
            print(f"\n✅ Categoría Real:     {categoria_real}")
            print(f"🤖 Categoría Predicha: {categoria_predicha}")
            
            correcto = "✅ CORRECTO" if categoria_real == categoria_predicha else "❌ INCORRECTO"
            print(f"   {correcto}")
            
            print(f"\n📊 Probabilidades:")
            for cat, prob in zip(self.categorias, probabilidades):
                barra = '█' * int(prob * 30)
                print(f"   {cat:<20} {prob:.4f} {barra}")
    
    def ejecutar_pipeline_completo(self):
        """Ejecuta el pipeline completo de entrenamiento y evaluación"""
        print("\n" + "🚀"*40)
        print("PIPELINE COMPLETO DE CLASIFICACIÓN DE NOTICIAS")
        print("🚀"*40 + "\n")
        
        # 1. Cargar y preprocesar datos
        self.cargar_datos_preprocesados()
        
        # 2. Entrenar modelo base
        self.entrenar_modelo_base()
        
        # 3. Evaluar modelo base
        metricas_base = self.evaluar_modelo(self.modelo, "Modelo Base")
        
        # 4. Validación cruzada del modelo base
        scores_base = self.validacion_cruzada(self.modelo, "Modelo Base")
        
        # 5. Optimizar hiperparámetros
        mejor_modelo = self.optimizar_hiperparametros()
        
        # 6. Evaluar modelo optimizado
        metricas_optimizado = self.evaluar_modelo(mejor_modelo, "Modelo Optimizado")
        
        # 7. Validación cruzada del modelo optimizado
        scores_optimizado = self.validacion_cruzada(mejor_modelo, "Modelo Optimizado")
        
        # 8. Visualizaciones
        print("\n" + "="*80)
        print("📊 GENERANDO VISUALIZACIONES")
        print("="*80)
        
        self.visualizar_matriz_confusion(metricas_base)
        self.visualizar_matriz_confusion(metricas_optimizado)
        self.visualizar_metricas_por_categoria(metricas_base)
        self.visualizar_metricas_por_categoria(metricas_optimizado)
        self.comparar_modelos_visualizacion(metricas_base, metricas_optimizado)
        
        # 9. Predicciones de ejemplo
        self.predecir_ejemplo(mejor_modelo, num_ejemplos=5)
        
        # 10. Guardar mejor modelo
        self.guardar_modelo(mejor_modelo, 'modelo_logistic_regression_optimizado.pkl')
        
        # Resumen final
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"\n📊 RESUMEN FINAL:")
        print(f"{'─'*80}")
        print(f"{'MODELO':<30} {'Train Acc':<12} {'Test Acc':<12} {'F1-Score':<12}")
        print(f"{'─'*80}")
        print(f"{'Modelo Base':<30} {metricas_base['train_accuracy']:<12.4f} {metricas_base['test_accuracy']:<12.4f} {metricas_base['f1_score']:<12.4f}")
        print(f"{'Modelo Optimizado':<30} {metricas_optimizado['train_accuracy']:<12.4f} {metricas_optimizado['test_accuracy']:<12.4f} {metricas_optimizado['f1_score']:<12.4f}")
        print(f"{'─'*80}")
        
        mejora = (metricas_optimizado['test_accuracy'] - metricas_base['test_accuracy']) * 100
        print(f"\n📈 Mejora en Test Accuracy: {mejora:+.2f}%")
        
        print(f"\n📁 Archivos generados:")
        print(f"   - modelo_logistic_regression_optimizado.pkl")
        print(f"   - matriz_confusion_modelo_base.png")
        print(f"   - matriz_confusion_modelo_optimizado.png")
        print(f"   - metricas_categoria_modelo_base.png")
        print(f"   - metricas_categoria_modelo_optimizado.png")
        print(f"   - comparacion_modelos.png")
        
        print("\n" + "🎉"*40)
        print("¡CLASIFICADOR DE NOTICIAS COMPLETADO!")
        print("🎉"*40 + "\n")


if __name__ == "__main__":
    # Crear instancia del clasificador
    clasificador = ClasificadorNoticias()
    
    # Ejecutar pipeline completo
    clasificador.ejecutar_pipeline_completo()