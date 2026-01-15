import streamlit as st
import requests
from bs4 import BeautifulSoup
from clasificador_noticias import ClasificadorNoticias
from preprocesamiento import PreprocesadorNoticias
import re

# Configuración de página
st.set_page_config(
    page_title="Clasificador de Noticias ABC",
    page_icon="📰",
    layout="centered"
)

# Inicializar clasificador
@st.cache_resource
def cargar_modelo():
    clasificador = ClasificadorNoticias()
    clasificador.cargar_modelo("modelo_randomforest.pkl")
    return clasificador

# Función de preprocesamiento (adaptada de tu clase)
def preprocesar_texto(texto):
    prep = PreprocesadorNoticias()
    return prep.limpiar_texto(texto)

# Scraping simple
def scrape_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extraer título
        titulo = soup.find('h1')
        titulo_texto = titulo.get_text().strip() if titulo else ""
        
        # Extraer subtítulo/descripción
        descripcion = ""
        # Intenta diferentes selectores comunes
        subtitulo = soup.find('h2', class_='voc-subtitle') or \
                    soup.find('p', class_='lead') or \
                    soup.find('div', class_='summary')
        
        if subtitulo:
            descripcion = subtitulo.get_text().strip()
        else:
            # Si no hay subtítulo, toma el primer párrafo
            primer_p = soup.find('p')
            if primer_p:
                descripcion = primer_p.get_text().strip()
        
        return f"{titulo_texto} {descripcion}"
    
    except Exception as e:
        raise Exception(f"Error al extraer contenido de la URL: {str(e)}")

# Detectar si es URL
def es_url(texto):
    return texto.startswith('http://') or texto.startswith('https://')

# CSS personalizado
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .stTextArea textarea {
        background-color: #1e293b;
        color: white;
        border: 1px solid #475569;
        border-radius: 10px;
    }
    .categoria-box {
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 24px;
    }
    .internacional { background: linear-gradient(135deg, #3b82f6, #2563eb); }
    .economia { background: linear-gradient(135deg, #10b981, #059669); }
    .tecnologia { background: linear-gradient(135deg, #a855f7, #9333ea); }
    .cultura { background: linear-gradient(135deg, #ec4899, #db2777); }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📰 Clasificador de Noticias ABC")
st.markdown("Ingresa una URL o texto para clasificar la noticia automáticamente")

# Cargar modelo
try:
    clasificador = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Input
input_text = st.text_area(
    "URL o Texto de la Noticia",
    height=150,
    placeholder="https://www.abc.es/... o pega el texto de la noticia aquí"
)

# Botón clasificar
if st.button("🔍 Clasificar Noticia", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Por favor, ingresa una URL o texto")
    else:
        with st.spinner("Clasificando..."):
            try:
                # Determinar si es URL o texto
                if es_url(input_text):
                    st.info("🌐 Detectada URL - Extrayendo contenido...")
                    texto_extraido = scrape_url(input_text)
                    texto_limpio = preprocesar_texto(texto_extraido)
                else:
                    st.info("📝 Procesando texto...")
                    texto_limpio = preprocesar_texto(input_text)
                
                # Clasificar
                resultado = clasificador.clasificar_con_probabilidades(texto_limpio)
                
                # Mostrar resultado
                st.success("✅ Clasificación completada")
                
                # Categoría predicha
                categoria = resultado['categoria']
                color_class = categoria.lower()
                
                st.markdown(f"""
                <div class="categoria-box {color_class}">
                    {categoria}
                </div>
                """, unsafe_allow_html=True)
                
                # Probabilidades
                st.subheader("📊 Probabilidades por Categoría")
                
                probs_sorted = sorted(
                    resultado['probabilidades'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for cat, prob in probs_sorted:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(prob, text=cat)
                    with col2:
                        st.write(f"{prob*100:.1f}%")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by Machine Learning | Categorías: Internacional, Economía, Tecnología, Cultura")