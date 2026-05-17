import pandas as pd
import numpy as np
import os

# 1. Definir la ruta de guardado relativa al script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_data = os.path.join(BASE_DIR, '..', 'data', 'datos_moda.csv')

def generar_dataset(n=1000):
    np.random.seed(42) # Para reproducibilidad
    
    # Generar estatura con distribución normal (media 170cm, std 10cm)
    estatura = np.random.normal(170, 10, n)
    estatura = np.clip(estatura, 145, 200) # Limitar entre 145cm y 200cm
    
    # Generar peso correlacionado con la estatura
    # Un IMC normal es entre 18.5 y 24.9. 
    # Añadimos un ruido para simular diferentes complexiones
    imc_base = np.random.normal(24, 4, n) # Media IMC 24, std 4
    
    # Peso = IMC * (Estatura en m)^2
    peso = imc_base * ((estatura/100)**2)
    peso = np.clip(peso, 40, 130) # Limitar entre 40kg y 130kg
    
    # Añadir un pequeño ruido extra al IMC calculado para que el modelo no 
    # aprenda solo la fórmula matemática, sino patrones más difusos
    
    tallas = []
    for i in range(n):
        # Calculamos el IMC real
        imc = peso[i] / ((estatura[i]/100)**2)
        
        # Le sumamos un poco de ruido para introducir variabilidad en la decisión de la talla
        # simulando que a veces a alguien le gusta más holgado o ajustado.
        imc_con_ruido = imc + np.random.normal(0, 1.5)
        
        if imc_con_ruido < 20: 
            tallas.append(0) # S
        elif imc_con_ruido < 25: 
            tallas.append(1) # M
        elif imc_con_ruido < 30: 
            tallas.append(2) # L
        elif imc_con_ruido < 35: 
            tallas.append(3) # XL
        elif imc_con_ruido < 40: 
            tallas.append(4) # XXL
        else: 
            tallas.append(5) # XXXL  
            
    # Redondeamos los valores
    estatura = np.round(estatura).astype(int)
    peso = np.round(peso, 1)

    # Creacion y guardado del DATAFRAME
    df = pd.DataFrame({
        'estatura_cm': estatura,
        'peso_kg': peso,
        'talla_label': tallas
    })
    
    # Si no existe una carpeta 'data' la crea
    os.makedirs(os.path.dirname(ruta_data), exist_ok=True)
    
    df.to_csv(ruta_data, index=False)
    print(f"✅ ¡Dataset creado exitosamente en: {ruta_data}!")
    print(df.head())

if __name__ == "__main__":
    generar_dataset()