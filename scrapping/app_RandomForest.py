import customtkinter as ctk
from tkinter import messagebox
import requests
from bs4 import BeautifulSoup
from clasificador_noticias import ClasificadorNoticias
from preprocesamiento import PreprocesadorNoticias
import threading

# Configuración
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ClasificadorApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Clasificador de Noticias ABC")
        self.root.geometry("900x800")
        
        # Cargar modelo
        try:
            self.clasificador = ClasificadorNoticias()
            self.clasificador.cargar_modelo("modelo_randomforest.pkl")
            self.prep = PreprocesadorNoticias()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
            self.root.quit()
        
        self.crear_interfaz()
    
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
        subtitulo.pack(pady=(0, 20))
        
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
        
        # Botón
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
        self.resultado_frame = ctk.CTkScrollableFrame(main_frame, height=350)
        self.resultado_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Footer
        footer = ctk.CTkLabel(
            main_frame,
            text="Powered by Machine Learning | Categorias: Internacional, Economia, Tecnologia, Cultura",
            font=ctk.CTkFont(size=10),
            text_color="#64748b"
        )
        footer.pack(pady=10)
    
    def clasificar_threaded(self):
        thread = threading.Thread(target=self.clasificar, daemon=True)
        thread.start()
    
    def clasificar(self):
        input_text = self.text_input.get("1.0", "end").strip()
        
        if not input_text:
            messagebox.showwarning("Advertencia", "Por favor, ingresa una URL o texto")
            return
        
        self.btn_clasificar.configure(state="disabled", text="CLASIFICANDO...")
        
        # Limpiar resultados
        for widget in self.resultado_frame.winfo_children():
            widget.destroy()
        
        try:
            # Procesar
            if input_text.startswith('http'):
                texto = self.scrape_url(input_text)
                texto_limpio = self.prep.limpiar_texto(texto)
            else:
                texto_limpio = self.prep.limpiar_texto(input_text)
            
            # Clasificar
            resultado = self.clasificador.clasificar_con_probabilidades(texto_limpio)
            
            # Mostrar
            self.mostrar_resultado(resultado)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
        
        finally:
            self.btn_clasificar.configure(state="normal", text="CLASIFICAR NOTICIA")
    
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
        
        # Categoría
        categoria = resultado['categoria']
        
        # Iconos y colores por categoría
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
        
        # Probabilidades
        probs_titulo = ctk.CTkLabel(
            self.resultado_frame,
            text="Probabilidades por Categoria",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        probs_titulo.pack(anchor="w", pady=(0, 15), padx=10)
        
        for cat, prob in sorted(resultado['probabilidades'].items(), key=lambda x: x[1], reverse=True):
            # Frame categoría
            cat_frame = ctk.CTkFrame(self.resultado_frame, fg_color="transparent")
            cat_frame.pack(fill="x", pady=8, padx=10)
            
            # Frame para nombre e icono
            header_frame = ctk.CTkFrame(cat_frame, fg_color="transparent")
            header_frame.pack(fill="x", pady=(0, 5))
            
            # Icono + Nombre
            icono_label = ctk.CTkLabel(
                header_frame,
                text=f"{iconos.get(cat, '')} {cat}",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            icono_label.pack(side="left")
            
            # Porcentaje
            porcentaje = ctk.CTkLabel(
                header_frame,
                text=f"{prob*100:.1f}%",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=colores.get(cat, "#64748b")
            )
            porcentaje.pack(side="right")
            
            # Barra de progreso con colores más brillantes
            progress = ctk.CTkProgressBar(
                cat_frame,
                height=12,
                corner_radius=6,
                progress_color=colores.get(cat, "#64748b"),
                fg_color="#334155"  # Fondo más claro para mejor contraste
            )
            progress.pack(fill="x")
            progress.set(prob)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ClasificadorApp()
    app.run()