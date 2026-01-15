import customtkinter as ctk
from tkinter import messagebox
import requests
from bs4 import BeautifulSoup
from clasificador_noticias import ClasificadorNoticias
from preprocesamiento import PreprocesadorNoticias
import threading
import numpy as np

# Configuración
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ClasificadorApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Clasificador de Noticias - Multi-Modelo")
        self.root.geometry("900x950")
        
        # Inicializar preprocesador
        self.prep = PreprocesadorNoticias()
        
        # Variables para modelos
        self.clasificador = None
        self.modelo_actual = "Logistic Regression"
        self.modelos_disponibles = {
            "Logistic Regression": "modelo_logistic_regression.pkl",
            "Random Forest": "modelo_randomforest.pkl"
        }
        
        # Variable para guardar texto original
        self.texto_original = ""
        
        # Cargar modelo inicial
        self.cargar_modelo_inicial()
        
        self.crear_interfaz()
    
    def cargar_modelo_inicial(self):
        """Intenta cargar el primer modelo disponible"""
        for nombre_modelo, ruta in self.modelos_disponibles.items():
            try:
                self.clasificador = ClasificadorNoticias()
                self.clasificador.cargar_modelo(ruta)
                self.modelo_actual = nombre_modelo
                print(f"✓ Modelo cargado: {nombre_modelo}")
                return
            except Exception as e:
                print(f"✗ No se pudo cargar {nombre_modelo}: {e}")
                continue
        
        messagebox.showerror("Error", "No se pudo cargar ningún modelo")
        self.root.quit()
    
    def cambiar_modelo(self, nombre_modelo):
        """Cambia el modelo activo"""
        if nombre_modelo == self.modelo_actual:
            return
        
        try:
            ruta = self.modelos_disponibles[nombre_modelo]
            self.clasificador = ClasificadorNoticias()
            self.clasificador.cargar_modelo(ruta)
            self.modelo_actual = nombre_modelo
            
            # Actualizar indicador visual
            self.actualizar_indicador_modelo()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
    
    def crear_interfaz(self):
        # Frame principal
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Título
        titulo = ctk.CTkLabel(
            main_frame,
            text="CLASIFICADOR DE NOTICIAS",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#60a5fa"
        )
        titulo.pack(pady=(20, 5))
        
        subtitulo = ctk.CTkLabel(
            main_frame,
            text="Ingresa una URL o texto para clasificar automaticamente",
            font=ctk.CTkFont(size=13),
            text_color="#94a3b8"
        )
        subtitulo.pack(pady=(0, 10))
        
        # Selector de modelo
        modelo_frame = ctk.CTkFrame(main_frame, fg_color="#1e293b", corner_radius=10)
        modelo_frame.pack(fill="x", padx=20, pady=10)
        
        modelo_label = ctk.CTkLabel(
            modelo_frame,
            text="🤖 Selecciona el Modelo",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        modelo_label.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Frame para botones de modelo
        botones_frame = ctk.CTkFrame(modelo_frame, fg_color="transparent")
        botones_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.botones_modelo = {}
        for nombre_modelo in self.modelos_disponibles.keys():
            btn = ctk.CTkButton(
                botones_frame,
                text=nombre_modelo,
                font=ctk.CTkFont(size=12, weight="bold"),
                height=40,
                command=lambda m=nombre_modelo: self.cambiar_modelo(m),
                fg_color="#3b82f6" if nombre_modelo == self.modelo_actual else "#475569",
                hover_color="#2563eb" if nombre_modelo == self.modelo_actual else "#64748b"
            )
            btn.pack(side="left", padx=5, expand=True, fill="x")
            self.botones_modelo[nombre_modelo] = btn
        
        # Indicador de modelo activo
        self.indicador_modelo = ctk.CTkLabel(
            modelo_frame,
            text=f"✓ Modelo activo: {self.modelo_actual}",
            font=ctk.CTkFont(size=11),
            text_color="#10b981"
        )
        self.indicador_modelo.pack(pady=(0, 10), padx=15, anchor="w")
        
        # Input frame
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        label = ctk.CTkLabel(
            input_frame,
            text="URL o Texto de la Noticia",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(anchor="w", padx=15, pady=(15, 5))
        
        self.text_input = ctk.CTkTextbox(
            input_frame,
            height=150,
            font=ctk.CTkFont(size=12)
        )
        self.text_input.pack(fill="both", padx=15, pady=(0, 15))
        
        # Botón clasificar
        self.btn_clasificar = ctk.CTkButton(
            input_frame,
            text="CLASIFICAR NOTICIA",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=50,
            command=self.clasificar_threaded,
            fg_color="#3b82f6",
            hover_color="#2563eb"
        )
        self.btn_clasificar.pack(fill="x", padx=15, pady=(0, 15))
        
        # Frame resultados
        self.resultado_frame = ctk.CTkScrollableFrame(main_frame, height=400)
        self.resultado_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Footer
        footer = ctk.CTkLabel(
            main_frame,
            text="Powered by Machine Learning | Categorias: Internacional, Economia, Tecnologia, Cultura",
            font=ctk.CTkFont(size=10),
            text_color="#64748b"
        )
        footer.pack(pady=10)
    
    def actualizar_indicador_modelo(self):
        """Actualiza los colores de los botones y el indicador"""
        for nombre, btn in self.botones_modelo.items():
            if nombre == self.modelo_actual:
                btn.configure(fg_color="#3b82f6", hover_color="#2563eb")
            else:
                btn.configure(fg_color="#475569", hover_color="#64748b")
        
        self.indicador_modelo.configure(text=f"✓ Modelo activo: {self.modelo_actual}")
    
    def clasificar_threaded(self):
        thread = threading.Thread(target=self.clasificar, daemon=True)
        thread.start()
    
    def obtener_palabras_importantes(self, texto_limpio, categoria=None, n_palabras=5):
        """Obtiene palabras importantes según el tipo de modelo"""
        try:
            if not hasattr(self.clasificador, 'vectorizer') or not hasattr(self.clasificador, 'modelo'):
                return self.obtener_palabras_simples(texto_limpio, n_palabras)
            
            X = self.clasificador.vectorizer.transform([texto_limpio])
            feature_names = self.clasificador.vectorizer.get_feature_names_out()
            palabras_texto = X.toarray()[0]
            indices_presentes = np.where(palabras_texto > 0)[0]
            
            if len(indices_presentes) == 0:
                return self.obtener_palabras_simples(texto_limpio, n_palabras)
            
            # Logistic Regression: usa coeficientes por categoría
            if hasattr(self.clasificador.modelo, 'coef_') and categoria:
                cat_idx = list(self.clasificador.modelo.classes_).index(categoria)
                coeficientes = self.clasificador.modelo.coef_[cat_idx]
                
                importancias = []
                for idx in indices_presentes:
                    importancia = coeficientes[idx] * palabras_texto[idx]
                    importancias.append((feature_names[idx], importancia))
                
                importancias.sort(key=lambda x: abs(x[1]), reverse=True)
                return [palabra for palabra, _ in importancias[:n_palabras]]
            
            # Random Forest: usa feature_importances globales
            elif hasattr(self.clasificador.modelo, 'feature_importances_'):
                importancias_modelo = self.clasificador.modelo.feature_importances_
                
                importancias = []
                for idx in indices_presentes:
                    importancia = importancias_modelo[idx] * palabras_texto[idx]
                    importancias.append((feature_names[idx], importancia))
                
                importancias.sort(key=lambda x: x[1], reverse=True)
                return [palabra for palabra, _ in importancias[:n_palabras]]
            
            else:
                return self.obtener_palabras_simples(texto_limpio, n_palabras)
                
        except Exception as e:
            print(f"Error al obtener palabras: {e}")
            import traceback
            traceback.print_exc()
            return self.obtener_palabras_simples(texto_limpio, n_palabras)
    
    def obtener_palabras_simples(self, texto_limpio, n_palabras=5):
        """Método alternativo: extrae las palabras más largas/relevantes del texto"""
        palabras = texto_limpio.split()
        palabras_filtradas = [p for p in palabras if len(p) > 4]
        palabras_unicas = []
        for p in palabras_filtradas:
            if p not in palabras_unicas:
                palabras_unicas.append(p)
        
        return palabras_unicas[:n_palabras] if palabras_unicas else ["texto", "noticia", "información"]
    
    def clasificar(self):
        input_text = self.text_input.get("1.0", "end").strip()
        
        if not input_text:
            self.root.after(0, lambda: messagebox.showwarning("Advertencia", "Por favor, ingresa una URL o texto"))
            return
        
        self.root.after(0, lambda: self.btn_clasificar.configure(state="disabled", text="CLASIFICANDO..."))
        self.root.after(0, self.limpiar_resultados)
        
        try:
            # Procesar
            if input_text.startswith('http'):
                texto = self.scrape_url(input_text)
                self.texto_original = texto  # Guardar texto original
                texto_limpio = self.prep.limpiar_texto(texto)
            else:
                self.texto_original = input_text  # Guardar texto original
                texto_limpio = self.prep.limpiar_texto(input_text)
            
            # Clasificar
            resultado = self.clasificador.clasificar_con_probabilidades(texto_limpio)
            
            # Obtener palabras importantes (siempre globales para ambos modelos)
            categoria_principal = resultado['categoria']
            resultado['palabras_clave'] = self.obtener_palabras_importantes(
                texto_limpio, categoria_principal, n_palabras=6
            )
            resultado['tipo_palabras'] = 'global'
            
            self.root.after(0, lambda: self.mostrar_resultado(resultado))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        finally:
            self.root.after(0, lambda: self.btn_clasificar.configure(state="normal", text="CLASIFICAR NOTICIA"))
    
    def limpiar_resultados(self):
        """Método para limpiar resultados de forma segura"""
        for widget in self.resultado_frame.winfo_children():
            widget.destroy()
    
    def scrape_url(self, url):
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        titulo = soup.find('h1')
        titulo_texto = titulo.get_text().strip() if titulo else ""
        
        subtitulo = soup.find('h2') or soup.find('p')
        descripcion = subtitulo.get_text().strip() if subtitulo else ""
        
        return f"{titulo_texto} {descripcion}"
    
    def mostrar_resultado(self, resultado):
        # Título
        titulo = ctk.CTkLabel(
            self.resultado_frame,
            text="Resultado de la Clasificacion",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        titulo.pack(pady=15)
        
        # NUEVO: Mostrar texto original extraído
        if self.texto_original:
            texto_frame = ctk.CTkFrame(
                self.resultado_frame, 
                fg_color="#1e293b", 
                corner_radius=10
            )
            texto_frame.pack(fill="x", padx=10, pady=(0, 20))
            
            texto_titulo = ctk.CTkLabel(
                texto_frame,
                text="📄 Texto Original Analizado",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            texto_titulo.pack(pady=(15, 10), padx=15, anchor="w")
            
            # Textbox para mostrar el texto (con scroll si es largo)
            texto_display = ctk.CTkTextbox(
                texto_frame,
                height=100,
                font=ctk.CTkFont(size=11),
                fg_color="#0f172a",
                wrap="word"
            )
            texto_display.pack(fill="x", padx=15, pady=(0, 15))
            texto_display.insert("1.0", self.texto_original)
            texto_display.configure(state="disabled")  # Solo lectura
        
        categoria = resultado['categoria']
        
        # Iconos y colores
        iconos = {
            'Internacional': '🌍',
            'Economia': '💰',
            'Tecnologia': '💻',
            'Cultura': '🎨'
        }
        
        colores = {
            'Internacional': "#3b82f6",
            'Economia': "#10b981",
            'Tecnologia': "#a855f7",
            'Cultura': "#ec4899"
        }
        
        # Categoría predicha
        cat_frame = ctk.CTkFrame(
            self.resultado_frame, 
            fg_color=colores.get(categoria, "#64748b"),
            corner_radius=15
        )
        cat_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        cat_label = ctk.CTkLabel(
            cat_frame,
            text=f"{iconos.get(categoria, '')} {categoria.upper()}",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="white"
        )
        cat_label.pack(pady=20)
        
        # Palabras clave (ahora siempre se muestra de forma global)
        if 'palabras_clave' in resultado and resultado['palabras_clave']:
            palabras_frame = ctk.CTkFrame(self.resultado_frame, fg_color="#1e293b", corner_radius=10)
            palabras_frame.pack(fill="x", padx=10, pady=(0, 20))
            
            palabras_titulo = ctk.CTkLabel(
                palabras_frame,
                text="🔑 Palabras Clave Detectadas",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            palabras_titulo.pack(pady=(15, 10), padx=15, anchor="w")
            
            tags_frame = ctk.CTkFrame(palabras_frame, fg_color="transparent")
            tags_frame.pack(fill="x", padx=15, pady=(0, 15))
            
            for palabra in resultado['palabras_clave']:
                palabra_tag = ctk.CTkFrame(
                    tags_frame,
                    fg_color=colores.get(categoria, "#64748b"),
                    corner_radius=8
                )
                palabra_tag.pack(side="left", padx=3, pady=2)
                
                palabra_text = ctk.CTkLabel(
                    palabra_tag,
                    text=palabra,
                    font=ctk.CTkFont(size=11),
                    text_color="white"
                )
                palabra_text.pack(padx=10, pady=5)
        
        # Probabilidades
        probs_titulo = ctk.CTkLabel(
            self.resultado_frame,
            text="Probabilidades por Categoria",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        probs_titulo.pack(anchor="w", pady=(0, 15), padx=10)
        
        for cat, prob in sorted(resultado['probabilidades'].items(), key=lambda x: x[1], reverse=True):
            cat_main_frame = ctk.CTkFrame(self.resultado_frame, fg_color="#1e293b", corner_radius=10)
            cat_main_frame.pack(fill="x", pady=8, padx=10)
            
            cat_inner_frame = ctk.CTkFrame(cat_main_frame, fg_color="transparent")
            cat_inner_frame.pack(fill="x", padx=12, pady=12)
            
            header_frame = ctk.CTkFrame(cat_inner_frame, fg_color="transparent")
            header_frame.pack(fill="x", pady=(0, 5))
            
            icono_label = ctk.CTkLabel(
                header_frame,
                text=f"{iconos.get(cat, '')} {cat}",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            icono_label.pack(side="left")
            
            porcentaje = ctk.CTkLabel(
                header_frame,
                text=f"{prob*100:.1f}%",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=colores.get(cat, "#64748b")
            )
            porcentaje.pack(side="right")
            
            progress = ctk.CTkProgressBar(
                cat_inner_frame,
                height=12,
                corner_radius=6,
                progress_color=colores.get(cat, "#64748b"),
                fg_color="#334155"
            )
            progress.pack(fill="x", pady=(0, 8))
            progress.set(prob)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ClasificadorApp()
    app.run()