import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

class RedditCommentScraper:
    def __init__(self):
        # Headers para simular un navegador real
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
        }
        self.comentarios = []
    
    def extraer_comentarios_post(self, url):
        """
        Extrae comentarios de un post específico de Reddit
        
        Args:
            url: URL del post (ej. https://www.reddit.com/r/technology/comments/...)
        """
        try:
            print(f"Obteniendo: {url}")
            response = requests.get(url, headers=self.headers)
            
            # Verificar si la petición fue exitosa
            if response.status_code != 200:
                print(f"Error: código {response.status_code}")
                return
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Reddit tiene diferentes estructuras según la versión (old.reddit vs new)
            # Probamos con la estructura de old.reddit (más fácil de parsear)
            
            # Buscar todos los comentarios
            comentarios_html = soup.find_all('div', class_='entry')
            
            print(f"Encontrados {len(comentarios_html)} comentarios")
            
            for comentario in comentarios_html:
                try:
                    # Extraer texto del comentario
                    texto_elem = comentario.find('div', class_='md')
                    if not texto_elem:
                        continue
                    texto = texto_elem.get_text(strip=True)
                    
                    # Extraer autor
                    autor_elem = comentario.find('a', class_='author')
                    autor = autor_elem.get_text(strip=True) if autor_elem else '[deleted]'
                    
                    # Extraer puntuación (score)
                    score_elem = comentario.find('span', class_='score unvoted')
                    if not score_elem:
                        score_elem = comentario.find('span', class_='score')
                    
                    score = 0
                    if score_elem:
                        score_text = score_elem.get_text(strip=True)
                        # Extraer número (puede ser "45 points" o "hidden")
                        # Buscar dígitos en el texto sin usar re
                        numeros = ''.join(c if c.isdigit() else ' ' for c in score_text).split()
                        if numeros:
                            score = int(numeros[0])
                    
                    # Extraer fecha
                    fecha_elem = comentario.find('time')
                    fecha = fecha_elem.get('datetime') if fecha_elem else 'unknown'
                    
                    # Guardar comentario
                    if texto and len(texto) > 10:  # Filtrar comentarios muy cortos
                        self.comentarios.append({
                            'texto': texto,
                            'autor': autor,
                            'score': score,
                            'fecha': fecha,
                            'url_post': url
                        })
                
                except Exception as e:
                    print(f"Error procesando comentario: {e}")
                    continue
        
        except Exception as e:
            print(f"Error en la petición: {e}")
    
    def scrape_subreddit(self, subreddit, num_posts=5):
        """
        Scrape comentarios de múltiples posts de un subreddit
        
        Args:
            subreddit: nombre del subreddit (ej. 'technology')
            num_posts: número de posts a procesar
        """
        # Usar old.reddit para estructura HTML más simple
        base_url = f"https://old.reddit.com/r/{subreddit}"
        
        try:
            print(f"Accediendo a r/{subreddit}...")
            response = requests.get(base_url, headers=self.headers)
            
            if response.status_code != 200:
                print(f"No se pudo acceder al subreddit")
                return
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Encontrar enlaces a posts
            posts = soup.find_all('a', class_='comments', limit=num_posts)
            
            print(f"Encontrados {len(posts)} posts para procesar\n")
            
            for i, post in enumerate(posts, 1):
                post_url = "https://old.reddit.com" + post['href']
                print(f"\n[{i}/{len(posts)}] Procesando post...")
                self.extraer_comentarios_post(post_url)
                
                # Delay entre peticiones 
                if i < len(posts):
                    print("Esperando 3 segundos...")
                    time.sleep(3)  # 3 segundos entre posts
        
        except Exception as e:
            print(f"Error scrapeando subreddit: {e}")
    
    def guardar_datos(self, filename='reddit_comments.csv'):
        """Guarda los comentarios en un CSV"""
        if not self.comentarios:
            print("No hay comentarios para guardar")
            return
        
        df = pd.DataFrame(self.comentarios)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\nGuardados {len(self.comentarios)} comentarios en '{filename}'")
        print(f"\nResumen:")
        print(f"   - Total comentarios: {len(df)}")
        print(f"   - Autores únicos: {df['autor'].nunique()}")
        print(f"   - Score promedio: {df['score'].mean():.1f}")
        
        return df



if __name__ == "__main__":
    # Crear scraper
    scraper = RedditCommentScraper()
    
    # Opción 1: Scrape un post específico
    # url_post = "https://old.reddit.com/r/technology/comments/..."
    # scraper.extraer_comentarios_post(url_post)
    
    # Opción 2: Scrape varios posts de un subreddit
    scraper.scrape_subreddit('technology', num_posts=3)  # Empieza con pocos posts
    
    # Guardar resultados
    df = scraper.guardar_datos('reddit_comments.csv')
    
    # Mostrar muestra
    if df is not None:
        print("\nPrimeros comentarios:")
        print(df.head())