import joblib
import os


class ClasificadorNoticias:
    """
    Clasificador de noticias que carga un modelo ya entrenado
    y permite clasificar nuevos titulares.

    Uso:
        clasificador = ClasificadorNoticias()
        clasificador.cargar_modelo("modelo_randomforest.pkl")
        resultado = clasificador.clasificar("Trump anuncia nuevos aranceles")
        print(resultado)  # -> "Internacional"
    """

    def __init__(self):
        self.modelo = None
        self.vectorizer = None
        self.categorias = None
        self.modelo_cargado = False

    def cargar_modelo(self, ruta_modelo='modelo_randomforest.pkl'):
        """
        Carga un modelo entrenado desde un archivo .pkl

        Args:
            ruta_modelo: Ruta al archivo .pkl del modelo
        """
        if not os.path.exists(ruta_modelo):
            raise FileNotFoundError(
                f"No se encuentra el modelo '{ruta_modelo}'.\n"
                f"Debes ejecutar primero 'entrenar_modelo.py' para generar el modelo."
            )

        paquete = joblib.load(ruta_modelo)

        self.modelo = paquete['modelo']
        self.vectorizer = paquete['vectorizer']
        self.categorias = paquete['categorias']
        self.modelo_cargado = True

        print(f"Modelo cargado: {ruta_modelo}")
        print(f"Categorias disponibles: {self.categorias}")

    def clasificar(self, texto):
        """
        Clasifica un texto y devuelve la categoria predicha.

        Args:
            texto: Titular o texto de noticia a clasificar

        Returns:
            str: Categoria predicha
        """
        if not self.modelo_cargado:
            raise RuntimeError("Debes cargar un modelo primero con cargar_modelo()")

        texto_vectorizado = self.vectorizer.transform([texto])
        categoria = self.modelo.predict(texto_vectorizado)[0]

        return categoria

    def clasificar_con_probabilidades(self, texto):
        """
        Clasifica un texto y devuelve la categoria con las probabilidades.

        Args:
            texto: Titular o texto de noticia a clasificar

        Returns:
            dict: Diccionario con 'categoria' y 'probabilidades'
        """
        if not self.modelo_cargado:
            raise RuntimeError("Debes cargar un modelo primero con cargar_modelo()")

        texto_vectorizado = self.vectorizer.transform([texto])
        categoria = self.modelo.predict(texto_vectorizado)[0]
        probabilidades = self.modelo.predict_proba(texto_vectorizado)[0]

        probs_dict = {cat: float(prob) for cat, prob in zip(self.categorias, probabilidades)}

        return {
            'categoria': categoria,
            'probabilidades': probs_dict
        }

    def clasificar_varios(self, textos):
        """
        Clasifica una lista de textos.

        Args:
            textos: Lista de titulares a clasificar

        Returns:
            list: Lista de categorias predichas
        """
        if not self.modelo_cargado:
            raise RuntimeError("Debes cargar un modelo primero con cargar_modelo()")

        textos_vectorizados = self.vectorizer.transform(textos)
        categorias = self.modelo.predict(textos_vectorizados)

        return list(categorias)

    def mostrar_prediccion(self, texto):
        """
        Clasifica un texto y muestra el resultado de forma visual.

        Args:
            texto: Titular o texto de noticia a clasificar
        """
        resultado = self.clasificar_con_probabilidades(texto)

        print("\n" + "="*70)
        print("CLASIFICACION DE NOTICIA")
        print("="*70)
        print(f"\nTexto: {texto}")
        print(f"\nCategoria predicha: {resultado['categoria']}")
        print(f"\nProbabilidades:")

        for cat, prob in sorted(resultado['probabilidades'].items(),
                               key=lambda x: x[1], reverse=True):
            barra = '#' * int(prob * 40)
            print(f"   {cat:<15} {prob:.4f} ({prob*100:5.1f}%) |{barra}")


if __name__ == "__main__":
    # Ejemplo de uso
    clasificador = ClasificadorNoticias()

    # Cargar el modelo (debe existir, ejecutar entrenar_modelo.py primero)
    try:
        clasificador.cargar_modelo("modelo_randomforest.pkl")

        # Ejemplos de clasificacion
        titulares_prueba = [
            "Trump anuncia nuevos aranceles contra China",
            "El Ibex 35 cierra con ganancias tras los datos del PIB",
            "Nueva exposicion de arte contemporaneo en el Museo del Prado",
            "Apple presenta su nuevo iPhone con inteligencia artificial"
        ]

        print("\n" + "="*70)
        print("PRUEBA DEL CLASIFICADOR DE NOTICIAS")
        print("="*70)

        for titular in titulares_prueba:
            clasificador.mostrar_prediccion(titular)

        # Tambien puedes clasificar de forma simple
        print("\n" + "="*70)
        print("CLASIFICACION SIMPLE")
        print("="*70)
        for titular in titulares_prueba:
            categoria = clasificador.clasificar(titular)
            print(f"\n'{titular[:50]}...' -> {categoria}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPrimero ejecuta: python entrenar_modelo.py")
