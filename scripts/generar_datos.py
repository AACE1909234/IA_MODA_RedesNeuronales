import pandas as pd
import numpy as np
import os

# 1. Definir la ruta de guardado
ruta_data = os.path.join('..', 'data', 'datos_moda.csv')

def generar_dataset(n=1000):
    np.random.seed(42) # Para que los resultados sean iguales siempre
    
    estatura = np.random.randint(150, 200, n) # Estatura en cm
    peso = np.random.randint(50, 110, n)      # Peso en kg
    
    tallas = []
    for i in range(n):
        imc = peso[i] / ((estatura[i]/100)**2)
        
        if imc < 20: 
            tallas.append(0) # S
        elif imc < 25: 
            tallas.append(1) # M
        elif imc < 30: 
            tallas.append(2) # L
        elif imc < 35: 
            tallas.append(3) # XL
        elif imc < 40: 
            tallas.append(4) # XXL
        else: 
            tallas.append(5) # XXXL  
            
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