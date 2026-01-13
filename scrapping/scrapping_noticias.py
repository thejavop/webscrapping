from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time

class ABCNewsScraper:
    def __init__(self):
        """Inicializa el scraper con configuración de Selenium"""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Opciones para acelerar carga y evitar timeouts
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-images')  # No cargar imágenes
        options.add_argument('--blink-settings=imagesEnabled=false')
        options.add_argument('--disk-cache-size=0')  # Sin caché
        options.add_argument('--media-cache-size=0')
        options.page_load_strategy = 'none'  # No espera nada, carga inmediato

        # Configurar el servicio de ChromeDriver con timeout
        from selenium.webdriver.chrome.service import Service
        service = Service()
        
        self.driver = webdriver.Chrome(options=options, service=service)
        
        # Configurar timeouts muy cortos
        self.driver.set_page_load_timeout(10)  # Solo 10 segundos
        self.driver.set_script_timeout(10)

        self.noticias = []
        self.articulos_sin_descripcion = []

        self.secciones = {
            'Internacional': 'https://www.abc.es/internacional/',
            'Economia': 'https://www.abc.es/economia/',
            'Tecnologia': 'https://www.abc.es/tecnologia/',
            'Cultura': 'https://www.abc.es/cultura/'
        }

    def aceptar_cookies(self):
        """Acepta el banner de cookies si aparece"""
        try:
            print("Buscando banner de cookies...")
            selectores_cookies = [
                "//button[contains(text(), 'Aceptar')]",
                "//button[contains(text(), 'Acepto')]",
                "//button[contains(text(), 'Accept')]",
                "//a[contains(text(), 'Aceptar')]",
                "//button[@id='didomi-notice-agree-button']"
            ]
            
            for selector in selectores_cookies:
                try:
                    boton_cookies = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    boton_cookies.click()
                    print("✓ Cookies aceptadas")
                    time.sleep(1)
                    return
                except:
                    continue
                    
            print("No se encontró banner de cookies")
        except Exception as e:
            print(f"Error con cookies: {e}")

    def scroll_pagina(self, scrolls=3):
        """Hace scroll para cargar más contenido dinámico"""
        print(f"Haciendo scroll ({scrolls} veces)...")
        for i in range(scrolls):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            print(f"   Scroll {i+1}/{scrolls}")
    
    def extraer_articulos(self, max_articulos=100):
        """Extrae títulos, descripciones y URLs de artículos de la página actual"""
        articulos_extraidos = []

        try:
            selectores = [
                "article",
                "div[class*='noticia']",
                "div[class*='article']",
                "div[class*='news']",
                "div[class*='story']",
                "div[data-test*='article']",
                "li[class*='item']"
            ]

            articulos_elementos = []
            for selector in selectores:
                try:
                    elementos = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elementos and len(elementos) > 5:
                        articulos_elementos = elementos
                        print(f"✓ Encontrados {len(elementos)} elementos con selector: {selector}")
                        break
                except:
                    continue

            if not articulos_elementos:
                print("⚠️  No se encontraron artículos con ningún selector")
                return articulos_extraidos
            
            self.driver.implicitly_wait(0)
            articulos_elementos = articulos_elementos[:max_articulos]
            print(f"Procesando {len(articulos_elementos)} elementos...")

            for idx, elemento in enumerate(articulos_elementos, 1):
                try:
                    if idx % 10 == 0:
                        print(f"   Procesados {idx}/{len(articulos_elementos)} elementos... ({len(articulos_extraidos)} artículos válidos)")

                    titulo = None
                    url = None
                    
                    # ← NUEVO: Buscar URL del artículo (normalmente en un enlace <a>)
                    try:
                        link_elem = elemento.find_element(By.CSS_SELECTOR, "a[href]")
                        url = link_elem.get_attribute('href')
                        # Asegurar que sea una URL completa
                        if url and not url.startswith('http'):
                            url = f"https://www.abc.es{url}"
                    except:
                        pass

                    # Buscar título
                    try:
                        for tag in ['h1', 'h2', 'h3', 'h4']:
                            try:
                                titulo_elem = elemento.find_element(By.TAG_NAME, tag)
                                titulo = titulo_elem.text.strip()
                                if titulo and len(titulo) > 10:
                                    break
                            except:
                                continue
                        
                        if not titulo:
                            titulo_elem = elemento.find_element(By.CSS_SELECTOR, "[class*='title'], [class*='titulo'], [class*='headline']")
                            titulo = titulo_elem.text.strip()
                    except:
                        pass

                    descripcion = None
                    try:
                        desc_elem = elemento.find_element(By.CSS_SELECTOR, "p, [class*='description'], [class*='descripcion'], [class*='summary'], [class*='sumario']")
                        descripcion = desc_elem.text.strip()
                    except:
                        pass

                    if titulo and len(titulo) > 10:
                        articulos_extraidos.append({
                            'titulo': titulo,
                            'descripcion': descripcion if descripcion else "",
                            'url_interna': url if url else ""  # Solo para uso interno
                        })

                except Exception as e:
                    continue

            print(f"   Procesamiento completado: {len(articulos_elementos)} elementos")

            # Eliminar duplicados
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
    
    def extraer_primer_parrafo(self, url):
        """Visita una URL y extrae el subtítulo del artículo"""
        max_intentos = 2
        
        for intento in range(max_intentos):
            try:
                if intento > 0:
                    print(f"   🔄 Reintento {intento + 1}/{max_intentos}...")
                else:
                    print(f"   Visitando: {url[:60]}...")
                
                try:
                    self.driver.get(url)
                    # Con page_load_strategy='none', cargamos y esperamos un poco
                    time.sleep(3)  # Dar tiempo a que cargue el contenido básico
                except Exception as timeout_error:
                    if intento == max_intentos - 1:
                        print(f"   ✗ Timeout después de {max_intentos} intentos")
                        return ""
                    continue
                
                # Buscar directamente el subtítulo de ABC.es
                try:
                    subtitulo = self.driver.find_element(By.CSS_SELECTOR, "h2.voc-subtitle")
                    texto = subtitulo.text.strip()
                    if texto and len(texto) > 10:
                        print(f"   ✓ Subtítulo extraído ({len(texto)} chars)")
                        return texto
                except:
                    pass  # No encontró, continuar con fallback
                
                # Fallback: buscar en otros selectores comunes
                selectores_parrafo = [
                    "article p",
                    "div[class*='article-body'] p",
                    "div[class*='content'] p",
                    "div[class*='texto'] p"
                ]
                
                for selector in selectores_parrafo:
                    try:
                        elementos = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for elem in elementos:
                            texto = elem.text.strip()
                            if texto and len(texto) > 50:
                                print(f"   ✓ Contenido extraído ({len(texto)} chars)")
                                return texto
                    except:
                        continue
                
                # Si no encontró nada en este intento, reintentar
                if intento < max_intentos - 1:
                    print(f"   ⚠️  No se encontró contenido, reintentando...")
                    continue
                
                print(f"   ✗ No se pudo extraer contenido")
                return ""
                
            except Exception as e:
                if intento == max_intentos - 1:
                    print(f"   ✗ Error: {str(e)[:100]}")
                return ""
        
        return ""
    
    def completar_descripciones(self):
        """Visita los artículos sin descripción y extrae el primer párrafo"""
        if not self.articulos_sin_descripcion:
            print("\n✓ Todos los artículos tienen descripción")
            return
        
        print(f"\n{'='*60}")
        print(f"COMPLETANDO DESCRIPCIONES")
        print(f"Artículos sin descripción: {len(self.articulos_sin_descripcion)}")
        print(f"{'='*60}\n")
        
        for idx, articulo_info in enumerate(self.articulos_sin_descripcion, 1):
            print(f"\n[{idx}/{len(self.articulos_sin_descripcion)}] {articulo_info['titulo'][:60]}...")
            
            if not articulo_info['url']:
                print(f"   ✗ Sin URL disponible")
                continue
            
            descripcion = self.extraer_primer_parrafo(articulo_info['url'])
            
            if descripcion:
                # Actualizar la descripción en self.noticias
                for noticia in self.noticias:
                    if noticia['titulo'] == articulo_info['titulo']:
                        noticia['descripcion'] = descripcion
                        break
            
            # Pequeña pausa entre requests para no saturar
            time.sleep(2)
        
        print(f"\n{'='*60}")
        print(f"✓ Descripciones completadas")
        print(f"{'='*60}")
    
    def scrapear_seccion(self, nombre_categoria, url, max_articulos=200):
        """Scrape una sección específica de ABC"""
        print(f"\n{'='*60}")
        print(f"Scrapeando: {nombre_categoria}")
        print(f"URL: {url}")
        print(f"{'='*60}")

        articulos_categoria = []
        pagina = 1
        max_paginas = 10

        try:
            while len(articulos_categoria) < max_articulos and pagina <= max_paginas:
                print(f"\n{'*'*40}")
                print(f"PÁGINA {pagina}/{max_paginas}")
                print(f"Artículos acumulados: {len(articulos_categoria)}")
                print(f"{'*'*40}")

                if pagina == 1:
                    self.driver.get(url)
                    time.sleep(3)
                    
                    if len(self.noticias) == 0:
                        self.aceptar_cookies()

                self.scroll_pagina(scrolls=5)

                articulos = self.extraer_articulos(max_articulos - len(articulos_categoria))
                
                print(f"\n>>> Extraídos de la página: {len(articulos)} artículos")
                
                print("\n📰 PRIMEROS 3 TÍTULOS:")
                for i, art in enumerate(articulos[:3], 1):
                    print(f"   {i}. {art['titulo'][:80]}...")

                nuevos = 0
                duplicados = 0
                
                for articulo in articulos:
                    if articulo['titulo'] not in [a['titulo'] for a in articulos_categoria]:
                        articulos_categoria.append(articulo)
                        nuevos += 1
                        
                        # ← NUEVO: Guardar artículos sin descripción para procesarlos después
                        if not articulo['descripcion'] and articulo['url_interna']:
                            self.articulos_sin_descripcion.append({
                                'titulo': articulo['titulo'],
                                'url': articulo['url_interna']
                            })
                    else:
                        duplicados += 1

                print(f"\n>>> Nuevos agregados: {nuevos}")
                print(f">>> Duplicados ignorados: {duplicados}")
                print(f">>> TOTAL ACUMULADO: {len(articulos_categoria)} artículos")

                if nuevos == 0:
                    print("\n⚠️  NO SE AGREGARON ARTÍCULOS NUEVOS - Fin de paginación")
                    break

                try:
                    print("\n🔄 Buscando botón 'Cargar más'...")
                    
                    load_more_button = None
                    
                    selectores_load_more = [
                        "div.voc-btn__container button",
                        "div.voc-btn__container a",
                        "button[class*='cargar']",
                        "button[class*='more']",
                        "a[class*='cargar']",
                        "a[class*='more']",
                        "//button[contains(text(), 'Cargar')]",
                        "//button[contains(text(), 'Ver más')]",
                        "//a[contains(text(), 'Cargar')]",
                        "//a[contains(text(), 'Ver más')]"
                    ]
                    
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    
                    for selector in selectores_load_more:
                        try:
                            if selector.startswith("//"):
                                load_more_button = self.driver.find_element(By.XPATH, selector)
                            else:
                                load_more_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                            
                            if load_more_button.is_displayed():
                                print(f"✓ Botón 'Cargar más' encontrado con: {selector}")
                                break
                            else:
                                load_more_button = None
                        except:
                            continue
                    
                    if load_more_button:
                        try:
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_more_button)
                            time.sleep(1)
                            
                            self.driver.execute_script("arguments[0].click();", load_more_button)
                            print(f"✓ Click en 'Cargar más' ejecutado")
                            
                            time.sleep(4)
                            
                            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(2)
                            
                            pagina += 1
                            print(f"✓ Página {pagina} cargada")
                            
                        except Exception as e:
                            print(f"Error haciendo click en botón: {e}")
                            break
                    else:
                        print("✗ No se encontró botón 'Cargar más' - fin de contenido")
                        break
                        
                except Exception as e:
                    print(f"Error en carga de más artículos: {e}")
                    break

            for articulo in articulos_categoria:
                articulo['categoria'] = nombre_categoria
                self.noticias.append(articulo)

            print(f"\n{'='*60}")
            print(f"RESUMEN {nombre_categoria}:")
            print(f"Total extraído: {len(articulos_categoria)} artículos")
            print(f"Páginas procesadas: {pagina}")
            print(f"Total acumulado: {len(self.noticias)} artículos")
            print(f"{'='*60}")

            time.sleep(2)

        except Exception as e:
            print(f"Error scrapeando {nombre_categoria}: {e}")
    
    def scrapear_todas_secciones(self, max_por_categoria=200):
        """Scrape todas las secciones definidas"""
        print("="*60)
        print("INICIANDO SCRAPING DE ABC.es")
        print("="*60)
        print(f"Objetivo: {max_por_categoria} artículos por categoría")
        print(f"Categorías: {list(self.secciones.keys())}\n")

        for categoria, url in self.secciones.items():
            self.scrapear_seccion(categoria, url, max_por_categoria)

        # ← NUEVO: Completar descripciones después de scrapear todas las secciones
        self.completar_descripciones()

        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETADO")
        print(f"Total de artículos: {len(self.noticias)}")
        print(f"{'='*60}")
    
    def guardar_datos(self, filename='abc_news.csv'):
        """Guarda los datos en un CSV"""
        if not self.noticias:
            print("No hay noticias para guardar")
            return None

        df = pd.DataFrame(self.noticias)

        # Eliminar la columna url_interna antes de guardar
        if 'url_interna' in df.columns:
            df = df.drop('url_interna', axis=1)

        df = df.drop_duplicates(subset=['titulo'])
        df = df[df['titulo'].str.len() > 10]

        df.to_csv(filename, index=False, encoding='utf-8')

        print(f"\n✓ Datos guardados en '{filename}'")
        print(f"\nRESUMEN FINAL:")
        print(f"   - Total artículos: {len(df)}")
        
        # ← NUEVO: Mostrar estadísticas de descripciones
        con_descripcion = df[df['descripcion'].str.len() > 0].shape[0]
        sin_descripcion = df[df['descripcion'].str.len() == 0].shape[0]
        print(f"   - Con descripción: {con_descripcion}")
        print(f"   - Sin descripción: {sin_descripcion}")
        
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
    scraper = ABCNewsScraper()

    try:
        scraper.scrapear_todas_secciones(max_por_categoria=200)
        df = scraper.guardar_datos('abc_news.csv')

        if df is not None:
            print("\n✓ Proceso completado exitosamente")
            print(f"✓ Archivo generado: abc_news.csv")

    except KeyboardInterrupt:
        print("\nScraping interrumpido por el usuario")

    except Exception as e:
        print(f"\nError general: {e}")
        import traceback
        traceback.print_exc()

    finally:
        scraper.cerrar()