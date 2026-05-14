import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Carga el archivo DATASET generado
ruta_data = os.path.join('..', 'data', 'datos_moda.csv')
data = pd.read_csv(ruta_data)

# Separamos entradas (X) de las etiquetas (y)
X = data[['estatura_cm', 'peso_kg']]
y = data['talla_label']

# Escalamos los datos para que la red aprenda más rápido
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividimos en 80% para entrenar y 20% para examen final
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# 3. Definir la Arquitectura de la Red Neuronal
model = keras.Sequential([
    # Capa de entrada con 16 neuronas
    keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    # Capa oculta para patrones complejos
    keras.layers.Dense(12, activation='relu'),
    # Capa de salida: 6 neuronas (S, M, L, XL, XXL, XXXL)
    keras.layers.Dense(6, activation='softmax')
])

# 4. Compilación
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Entrenamiento (El momento del aprendizaje)
print("Estudiando datos...")
history = model.fit(X_train, y_train, epochs=60, validation_split=0.2, verbose=1)

# 6. Evaluación
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nEntrenamiento completado. Precisión: {accuracy*100:.2f}%")

# 7. Guardar el entrenamiento actual
ruta_modelo = os.path.join('..', 'models', 'modelo_moda.h5')
os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
model.save(ruta_modelo)
print(f"Modelo guardado en: {ruta_modelo}")