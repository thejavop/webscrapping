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
        options.add_argument('--headless')
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
            'Art': 'https://www.bbc.com/arts',
            'Culture': 'https://www.bbc.com/culture'
        }

    def aceptar_cookies(self):
        """Acepta el banner de cookies si aparece"""
        try:
            print("Buscando banner de cookies...")
            boton_cookies = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'agree')]"))
            )
            boton_cookies.click()
            print("Cookies aceptadas")
            time.sleep(1)
        except TimeoutException:
            print("No se encontró banner de cookies")
        except Exception as e:
            print(f"Error con cookies: {e}")

    def scroll_pagina(self, scrolls=3):
        """Hace scroll para cargar más contenido dinámico"""
        print(f"Haciendo scroll ({scrolls} veces)...")
        for i in range(scrolls):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            print(f"   Scroll {i+1}/{scrolls}")
    
    def extraer_articulos(self, max_articulos=100):
        """Extrae títulos y descripciones de artículos de la página actual"""
        articulos_extraidos = []

        try:
            contenedor = None
            try:
                contenedor = self.driver.find_element(By.CSS_SELECTOR, "#main-content > article")
                print("✓ Encontrado contenedor Latest Updates")
            except:
                print("⚠️  No se encontró el contenedor específico, usando página completa")
            
            # Buscar artículos dentro del contenedor (o en toda la página si no existe)
            if contenedor:
                base_element = contenedor
            else:
                base_element = self.driver

            selectores = [
                "div[data-testid='anchor-inner-wrapper']",
                "div[class*='promo']",
                "article",
                "div[class*='gel-layout__item']"
            ]

            articulos_elementos = []
            for selector in selectores:
                try:
                    elementos = base_element.find_elements(By.CSS_SELECTOR, selector)
                    if elementos:
                        articulos_elementos = elementos
                        print(f"Encontrados {len(elementos)} elementos con selector: {selector}")
                        break
                except:
                    continue

            if not articulos_elementos:
                print("No se encontraron artículos con ningún selector")
                return articulos_extraidos
            
            self.driver.implicitly_wait(0)

            articulos_elementos = articulos_elementos[:max_articulos]
            print(f"Procesando {len(articulos_elementos)} elementos...")

            for idx, elemento in enumerate(articulos_elementos, 1):
                try:
                    if idx % 10 == 0:
                        print(f"   Procesados {idx}/{len(articulos_elementos)} elementos... ({len(articulos_extraidos)} artículos válidos)")

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

                    descripcion = None
                    try:
                        desc_elem = elemento.find_element(By.CSS_SELECTOR, "p, [class*='description'], [class*='summary']")
                        descripcion = desc_elem.text.strip()
                    except:
                        pass

                    if titulo and len(titulo) > 10:
                        articulos_extraidos.append({
                            'titulo': titulo,
                            'descripcion': descripcion if descripcion else ""
                        })

                except Exception as e:
                    continue

            print(f"   Procesamiento completado: {len(articulos_elementos)} elementos")

            titulos_vistos = set()
            articulos_unicos = []
            for articulo in articulos_extraidos:
                if articulo['titulo'] not in titulos_vistos:
                    titulos_vistos.add(articulo['titulo'])
                    articulos_unicos.append(articulo)

            return articulos_unicos[:max_articulos]

        except Exception as e:
            print(f"Error extrayendo artículos: {e}")
            return articulos_extraidos
    
    def scrapear_seccion(self, nombre_categoria, url, max_articulos=100):
        """Scrape una sección específica de BBC News"""
        print(f"\n{'='*60}")
        print(f"Scrapeando: {nombre_categoria}")
        print(f"URL: {url}")
        print(f"{'='*60}")

        articulos_categoria = []
        pagina = 1
        max_paginas = 11

        try:
            while len(articulos_categoria) < max_articulos and pagina <= max_paginas:
                print(f"\nPágina {pagina}/{max_paginas}")
                if pagina==1:
                    self.driver.get(url)
                    time.sleep(3)

                if len(self.noticias) == 0 and pagina == 1:
                    self.aceptar_cookies()

                self.scroll_pagina(scrolls=8)

                articulos = self.extraer_articulos(max_articulos - len(articulos_categoria))
                print(f"\n>>> Extraídos de la página: {len(articulos)} artículos")
                
                # Contar duplicados
                nuevos = 0
                duplicados = 0

                for articulo in articulos:
                    if articulo['titulo'] not in [a['titulo'] for a in articulos_categoria]:
                        articulos_categoria.append(articulo)
                        nuevos += 1
                    else:
                        duplicados += 1

                print(f"\n>>> Nuevos agregados: {nuevos}")
                print(f">>> Duplicados ignorados: {duplicados}")
                print(f">>> TOTAL ACUMULADO en articulos_categoria: {len(articulos_categoria)}")

                # MOSTRAR TOTAL EN MEMORIA
                print(f">>> TOTAL EN self.noticias: {len(self.noticias)}")
                try:
                    print("\nBuscando botón Next Page...")
                    
                    # Buscar dentro del contenedor de Latest Updates
                    contenedor = self.driver.find_element(By.CSS_SELECTOR, "#main-content > article > section.sc-8687269a-1.jXcOGh")
                    
                    next_button = None
                    
                    # Buscar el botón
                    try:
                        next_button = contenedor.find_element(By.XPATH, ".//button[@aria-label='Next Page']")
                        print("✓ Encontrado botón Next Page")
                    except:
                        try:
                            # Buscar por texto ">"
                            next_button = contenedor.find_element(By.XPATH, ".//button[contains(text(), '>')]")
                            print("✓ Encontrado por texto '>'")
                        except:
                            pass
                    
                    if next_button:
                        # Verificar si el botón está deshabilitado
                        if next_button.get_attribute('aria-disabled') == 'true':
                            print("✗ Botón deshabilitado - fin de páginas")
                            break
                        
                        # Capturar el texto del primer artículo ANTES del click
                        try:
                            primer_articulo_antes = contenedor.find_element(By.CSS_SELECTOR, "div[data-testid='anchor-inner-wrapper'] h3, div[data-testid='anchor-inner-wrapper'] h2").text
                        except:
                            primer_articulo_antes = None
                        
                        # Scroll hasta el botón
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                        time.sleep(1)
                        
                        # Click con JavaScript
                        self.driver.execute_script("arguments[0].click();", next_button)
                        print("Click ejecutado, esperando cambio...")
                        
                        # Esperar a que el contenido cambie (máximo 10 segundos)
                        cambio_detectado = False
                        for intento in range(10):
                            time.sleep(1)
                            try:
                                primer_articulo_despues = contenedor.find_element(By.CSS_SELECTOR, "div[data-testid='anchor-inner-wrapper'] h3, div[data-testid='anchor-inner-wrapper'] h2").text
                                
                                if primer_articulo_despues != primer_articulo_antes:
                                    print(f"✓ CONTENIDO CAMBIÓ en {intento+1} segundos")
                                    cambio_detectado = True
                                    pagina += 1
                                    break
                            except:
                                pass
                        
                        if not cambio_detectado:
                            print("✗ No se detectó cambio en el contenido después de 10 segundos")
                            break
                        
                        # Scroll adicional para cargar el contenido nuevo
                        self.scroll_pagina(scrolls=3)
                        
                    else:
                        print("✗ No se encontró botón Next Page")
                        break
                        
                except Exception as e:
                    print(f"Error en paginación: {e}")
                    break
                    

            for articulo in articulos_categoria:
                articulo['categoria'] = nombre_categoria
                self.noticias.append(articulo)

            print(f"Extraídos {len(articulos_categoria)} artículos de {nombre_categoria}")
            print(f"Total acumulado: {len(self.noticias)} artículos")

            time.sleep(1)

        except Exception as e:
            print(f"Error scrapeando {nombre_categoria}: {e}")
    
    def scrapear_todas_secciones(self, max_por_categoria=100):
        """Scrape todas las secciones definidas"""
        print("Iniciando scraping de BBC News...")
        print(f"Objetivo: {max_por_categoria} artículos por categoría")
        print(f"Categorías: {list(self.secciones.keys())}\n")

        for categoria, url in self.secciones.items():
            self.scrapear_seccion(categoria, url, max_por_categoria)

        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETADO")
        print(f"Total de artículos: {len(self.noticias)}")
        print(f"{'='*60}")
    
    def guardar_datos(self, filename='bbc_news.csv'):
        """Guarda los datos en un CSV"""
        if not self.noticias:
            print("No hay noticias para guardar")
            return None

        df = pd.DataFrame(self.noticias)

        df = df.drop_duplicates(subset=['titulo'])
        df = df[df['titulo'].str.len() > 10]

        df.to_csv(filename, index=False, encoding='utf-8')

        print(f"\nDatos guardados en '{filename}'")
        print(f"\nResumen:")
        print(f"   - Total artículos: {len(df)}")
        print(f"   - Distribución por categoría:")
        print(df['categoria'].value_counts().to_string())
        print(f"\nMuestra de datos:")
        print(df.head(3).to_string())

        return df

    def cerrar(self):
        """Cierra el navegador"""
        print("\nCerrando navegador...")
        self.driver.quit()


if __name__ == "__main__":
    scraper = BBCNewsScraper()

    try:
        scraper.scrapear_todas_secciones(max_por_categoria=200)
        df = scraper.guardar_datos('bbc_news.csv')

        if df is not None:
            print("\nProceso completado exitosamente")
            print(f"Archivo generado: bbc_news.csv")

    except KeyboardInterrupt:
        print("\nScraping interrumpido por el usuario")

    except Exception as e:
        print(f"\nError general: {e}")

    finally:
        scraper.cerrar()