import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
import os

# 1. Configuracion de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(BASE_DIR, '..', 'models', 'modelo_moda.keras')
ruta_scaler = os.path.join(BASE_DIR, '..', 'models', 'scaler_moda.joblib')

# 2. Carga del modelo, escalador y configuracion de etiquetas
try:
    model = keras.models.load_model(ruta_modelo)
    scaler = joblib.load(ruta_scaler)
except Exception as e:
    print("Error: No se encontró el modelo o el escalador. Por favor, entrena la IA primero.")
    print(f"Detalles del error: {e}")
    exit()

tallas_nombres = ["Small (S)", "Medium (M)", "Large (L)", "Extra Large (XL)", "XXL", "XXXL"]

def sugerir_corte(imc, preferencia):
    # Logica de negocio para determinacion de tipo de corte
    if preferencia == 1: # Ajustado
        return "Slim Fit"
    elif preferencia == 3: # Holgado
        return "Oversize"
    else:
        if imc > 28:
            return "Boxy Fit (Corte cuadrado y holgado)"
        else:
            return "Regular Fit"

def obtener_input_numerico(mensaje):
    while True:
        try:
            valor = float(input(mensaje))
            if valor <= 0:
                print("Por favor, ingresa un número mayor a cero.")
                continue
            return valor
        except ValueError:
            print("Error: Ingresa únicamente un número válido.")

def obtener_input_opcion(mensaje, opciones_validas):
    while True:
        try:
            valor = int(input(mensaje))
            if valor in opciones_validas:
                return valor
            else:
                print(f"Por favor, selecciona una opción válida {opciones_validas}.")
        except ValueError:
            print("Error: Ingresa un número entero.")

def ejecutar_sistema():
    print("--- SISTEMA DE RECOMENDACION DE MODA (ANN) ---")
    print("Facultad de Ingenieria Mecanica y Electrica - UANL\n")
    
    estatura = obtener_input_numerico("Introduce tu estatura en centimetros (ej. 175): ")
    peso = obtener_input_numerico("Introduce tu peso en kilogramos (ej. 75): ")
    
    print("\nSelecciona tu preferencia de ajuste:")
    print("1. Ajustado")
    print("2. Normal")
    print("3. Holgado")
    pref = obtener_input_opcion("Opcion (1-3): ", [1, 2, 3])
    
    # 3. Procesamiento de datos de usuario (usando el escalador entrenado)
    # Usamos DataFrame en lugar de numpy array para evitar advertencias de scikit-learn
    usuario_data = pd.DataFrame([[estatura, peso]], columns=['estatura_cm', 'peso_kg'])
    usuario_scaled = scaler.transform(usuario_data)
    
    # 4. Inferencia con la Red Neuronal
    prediccion = model.predict(usuario_scaled, verbose=0)
    clase_predicha = int(np.argmax(prediccion))
    confianza = float(np.max(prediccion)) * 100
    
    # 5. Determinacion de estilo
    imc_usuario = peso / ((estatura/100)**2)
    corte_sugerido = sugerir_corte(imc_usuario, pref)
    
    print("\n" + "="*40)
    print("RESULTADOS DEL ANALISIS")
    print("="*40)
    print(f"Talla recomendada: {tallas_nombres[clase_predicha]}")
    print(f"Corte sugerido: {corte_sugerido}")
    print(f"Nivel de confianza del modelo: {confianza:.2f}%")
    print("="*40)

if __name__ == "__main__":
    ejecutar_sistema()