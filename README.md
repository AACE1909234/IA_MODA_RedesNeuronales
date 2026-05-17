# IA_MODA_RedesNeuronales
IA generada en PYTHON para ayuda en la selección de talla y corte según tu peso y estatura
# Sistema de Recomendacion de Moda Basado en Redes Neuronales Artificiales

## Descripcion
Este proyecto consiste en el desarrollo de una Inteligencia Artificial diseñada para optimizar la seleccion de tallas y cortes de ropa en entornos de comercio electronico. Utiliza una arquitectura de Red Neuronal Artificial (ANN) para procesar datos antropometricos y devolver recomendaciones precisas, alineandose con los objetivos de reduccion de devoluciones y personalizacion de la experiencia de compra.

## Estructura del Proyecto
* data/: Almacenamiento del dataset en formato CSV.
* models/: Archivos de pesos y sesgos del modelo entrenado (.h5).
* scripts/: Codigo fuente para generacion de datos, entrenamiento y prediccion.

## Gestion del Dataset (Alimentacion de la IA)
El sistema permite dos metodos para gestionar los datos de entrenamiento:

### Opcion A: Generar Dataset Aleatorio
Si no se cuenta con datos reales, se puede ejecutar el script de simulacion:
python scripts/generar_datos.py
Este comando generara 1000 registros basados en distribuciones estadisticas de peso y estatura.

### Opcion B: Uso de Dataset Propio
Para utilizar un DataFrame personalizado:
1. Asegurese de que su archivo se nombre 'datos_moda.csv'.
2. El archivo debe contener las columnas: 'estatura_cm', 'peso_kg' y 'talla_label'.
3. Coloque el archivo dentro de la carpeta 'data/'.
4. Ejecute el script de entrenamiento para que la IA aprenda de sus datos especificos.

## Personalizacion de Cortes (Boxy Fit, Oversize, Slim Fit)
El modelo ha sido diseñado para integrar recomendaciones de estilo. Para modificar o añadir tipos de corte:
1. En el script de generacion, incluya la variable de estilo deseada.
2. La IA procesa la relacion entre las dimensiones fisicas y el volumen de la prenda para sugerir:
   - Slim Fit: Ajuste entallado.
   - Regular Fit: Ajuste estandar.
   - Boxy Fit: Corte cuadrado y corto.
   - Oversize: Corte amplio y holgado.

## Requisitos del Sistema
* Python 3.8+
* TensorFlow 2.x
* Pandas
* Scikit-learn

## Instalacion y Uso
1. Instalar dependencias: pip install -r requirements.txt
2. Entrenar el modelo: python scripts/entrenar_ia.py
3. Ejecutar recomendador: python scripts/recomendar.py

Desarrollado por: Angel Ariel Castillo Estrada
Matricula: 1909234
Institucion: Universidad Autonoma de Nuevo Leon
Facultad: Facultad de Ingenieria Mecanica y Electrica
