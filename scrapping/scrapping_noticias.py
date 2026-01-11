from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time

class BBCNewsScraper:
    def __init__(self):
        """Inicializa el scraper con configuración de Selenium"""
        # Configurar opciones de Chrome
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Ejecutar sin ventana (opcional)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Inicializar driver
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(10)
        
        # Almacenar noticias
        self.noticias = []
        
        # Definir secciones a scrape
        self.secciones = {
            'World': 'https://www.bbc.com/news/world',
            'Business': 'https://www.bbc.com/news/business',
            'Technology': 'https://www.bbc.com/news/technology',
            'Science': 'https://www.bbc.com/news/science-environment'
        }
    
    def aceptar_cookies(self):
        """Acepta el banner de cookies si aparece"""
        try:
            print("🍪 Buscando banner de cookies...")
            # Esperar y hacer clic en botón de aceptar cookies
            boton_cookies = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'agree')]"))
            )
            boton_cookies.click()
            print("✅ Cookies aceptadas")
            time.sleep(2)
        except TimeoutException:
            print("ℹ️ No se encontró banner de cookies")
        except Exception as e:
            print(f"⚠️ Error con cookies: {e}")
    
    def scroll_pagina(self, scrolls=3):
        """Hace scroll para cargar más contenido dinámico"""
        print(f"📜 Haciendo scroll ({scrolls} veces)...")
        for i in range(scrolls):
            # Scroll hasta el final
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Esperar a que cargue contenido
            print(f"   Scroll {i+1}/{scrolls}")
    
    def extraer_articulos(self, max_articulos=100):
        """Extrae títulos y descripciones de artículos de la página actual"""
        articulos_extraidos = []

        try:
            # Selector para artículos de BBC News
            # Nota: BBC usa diferentes estructuras, probamos varios selectores

            # Opción 1: Selectores comunes de BBC
            selectores = [
                "div[data-testid='anchor-inner-wrapper']",
                "div[class*='promo']",
                "article",
                "div[class*='gel-layout__item']"
            ]

            articulos_elementos = []
            for selector in selectores:
                try:
                    elementos = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elementos:
                        articulos_elementos = elementos
                        print(f"✅ Encontrados {len(elementos)} elementos con selector: {selector}")
                        break
                except:
                    continue

            if not articulos_elementos:
                print("❌ No se encontraron artículos con ningún selector")
                return articulos_extraidos

            # Limitar al número máximo de artículos
            articulos_elementos = articulos_elementos[:max_articulos]
            print(f"⚙️ Procesando {len(articulos_elementos)} elementos...")

            for idx, elemento in enumerate(articulos_elementos, 1):
                try:
                    # Mostrar progreso cada 10 elementos
                    if idx % 10 == 0:
                        print(f"   Procesados {idx}/{len(articulos_elementos)} elementos... ({len(articulos_extraidos)} artículos válidos)")

                    # Intentar extraer título
                    titulo = None
                    try:
                        titulo_elem = elemento.find_element(By.CSS_SELECTOR, "h3, h2, [class*='title']")
                        titulo = titulo_elem.text.strip()
                    except:
                        try:
                            titulo_elem = elemento.find_element(By.TAG_NAME, "a")
                            titulo = titulo_elem.get_attribute("aria-label") or titulo_elem.text.strip()
                        except:
                            pass

                    # Intentar extraer descripción
                    descripcion = None
                    try:
                        desc_elem = elemento.find_element(By.CSS_SELECTOR, "p, [class*='description'], [class*='summary']")
                        descripcion = desc_elem.text.strip()
                    except:
                        pass

                    # Guardar si tenemos al menos el título
                    if titulo and len(titulo) > 10:
                        articulos_extraidos.append({
                            'titulo': titulo,
                            'descripcion': descripcion if descripcion else ""
                        })

                except Exception as e:
                    continue

            print(f"   ✅ Procesamiento completado: {len(articulos_elementos)} elementos")

            # Eliminar duplicados basados en título
            titulos_vistos = set()
            articulos_unicos = []
            for articulo in articulos_extraidos:
                if articulo['titulo'] not in titulos_vistos:
                    titulos_vistos.add(articulo['titulo'])
                    articulos_unicos.append(articulo)

            return articulos_unicos[:max_articulos]

        except Exception as e:
            print(f"❌ Error extrayendo artículos: {e}")
            return articulos_extraidos
    
    def scrape_seccion(self, nombre_categoria, url, max_articulos=100):
        """Scrape una sección específica de BBC News"""
        print(f"\n{'='*60}")
        print(f"📰 Scrapeando: {nombre_categoria}")
        print(f"🔗 URL: {url}")
        print(f"{'='*60}")
        
        try:
            # Navegar a la URL
            self.driver.get(url)
            time.sleep(3)
            
            # Aceptar cookies en la primera visita
            if len(self.noticias) == 0:
                self.aceptar_cookies()
            
            # Hacer scroll para cargar más artículos
            self.scroll_pagina(scrolls=5)
            
            # Extraer artículos
            articulos = self.extraer_articulos(max_articulos)
            
            # Añadir categoría a cada artículo
            for articulo in articulos:
                articulo['categoria'] = nombre_categoria
                self.noticias.append(articulo)
            
            print(f"✅ Extraídos {len(articulos)} artículos de {nombre_categoria}")
            print(f"📊 Total acumulado: {len(self.noticias)} artículos")
            
            # Delay entre secciones
            time.sleep(3)
        
        except Exception as e:
            print(f"❌ Error scrapeando {nombre_categoria}: {e}")
    
    def scrape_todas_secciones(self, max_por_categoria=100):
        """Scrape todas las secciones definidas"""
        print("🚀 Iniciando scraping de BBC News...")
        print(f"🎯 Objetivo: {max_por_categoria} artículos por categoría")
        print(f"📋 Categorías: {list(self.secciones.keys())}\n")
        
        for categoria, url in self.secciones.items():
            self.scrape_seccion(categoria, url, max_por_categoria)
        
        print(f"\n{'='*60}")
        print(f"✅ SCRAPING COMPLETADO")
        print(f"📊 Total de artículos: {len(self.noticias)}")
        print(f"{'='*60}")
    
    def guardar_datos(self, filename='bbc_news.csv'):
        """Guarda los datos en un CSV"""
        if not self.noticias:
            print("❌ No hay noticias para guardar")
            return None
        
        df = pd.DataFrame(self.noticias)
        
        # Eliminar filas con títulos duplicados
        df = df.drop_duplicates(subset=['titulo'])
        
        # Eliminar filas con títulos vacíos
        df = df[df['titulo'].str.len() > 10]
        
        # Guardar
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"\n💾 Datos guardados en '{filename}'")
        print(f"\n📊 Resumen:")
        print(f"   - Total artículos: {len(df)}")
        print(f"   - Distribución por categoría:")
        print(df['categoria'].value_counts().to_string())
        print(f"\n📝 Muestra de datos:")
        print(df.head(3).to_string())
        
        return df
    
    def cerrar(self):
        """Cierra el navegador"""
        print("\n🔒 Cerrando navegador...")
        self.driver.quit()


# 🚀 EJEMPLO DE USO
if __name__ == "__main__":
    # Crear scraper
    scraper = BBCNewsScraper()
    
    try:
        # Scrape todas las secciones (100 artículos por categoría)
        scraper.scrape_todas_secciones(max_por_categoria=100)
        
        # Guardar datos
        df = scraper.guardar_datos('bbc_news.csv')
        
        # Verificar datos
        if df is not None:
            print("\n✅ Proceso completado exitosamente")
            print(f"📁 Archivo generado: bbc_news.csv")
    
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrumpido por el usuario")
    
    except Exception as e:
        print(f"\n❌ Error general: {e}")
    
    finally:
        # Siempre cerrar el navegador
        scraper.cerrar()