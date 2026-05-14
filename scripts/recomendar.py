import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# 1. Configuracion de rutas
ruta_modelo = os.path.join('..', 'models', 'modelo_moda.h5')
ruta_data = os.path.join('..', 'data', 'datos_moda.csv')

# 2. Carga del modelo y configuracion de etiquetas
try:
    model = keras.models.load_model(ruta_modelo)
except Exception as e:
    print("Error: No se encontro el archivo del modelo en la ruta especificada.")
    print(f"Detalle: {e}")
    exit()

tallas_nombres = ["Small", "Medium", "Large", "Extra Large", "XXL", "XXXL"]

# Ajuste del Scaler para normalizacion de datos de entrada
data = pd.read_csv(ruta_data)
scaler = StandardScaler()
scaler.fit(data[['estatura_cm', 'peso_kg']])

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

def ejecutar_sistema():
    print("--- SISTEMA DE RECOMENDACION DE MODA (ANN) ---")
    print("Facultad de Ingenieria Mecanica y Electrica - UANL\n")
    
    try:
        estatura = float(input("Introduce tu estatura en centimetros (ej. 175): "))
        peso = float(input("Introduce tu peso en kilogramos (ej. 75): "))
        
        print("\nSelecciona tu preferencia de ajuste:")
        print("1. Ajustado")
        print("2. Normal")
        print("3. Holgado")
        pref = int(input("Opcion (1-3): "))
        
        # 3. Procesamiento de datos de usuario
        usuario_data = np.array([[estatura, peso]])
        usuario_scaled = scaler.transform(usuario_data)
        
        # 4. Inferencia con la Red Neuronal
        prediccion = model.predict(usuario_scaled, verbose=0)
        clase_predicha = np.argmax(prediccion)
        confianza = np.max(prediccion) * 100
        
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
        
    except ValueError:
        print("\nError: Ingrese unicamente valores numericos validos.")

if __name__ == "__main__":
    ejecutar_sistema()
    input("\nPresione Enter para finalizar el programa...")