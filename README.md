# Fashion MNIST Classifier

Este proyecto utiliza TensorFlow para entrenar un modelo en el dataset de Fashion MNIST. El modelo se entrena en Google Cloud Vertex AI y se puede implementar localmente o en la nube.

## Requisitos

- Python 3.10
- TensorFlow 2.16
- Google Cloud SDK

## Instalación

1. Clona este repositorio:
git clone 
https://github.com/jorgeahmed/fashion-mnist-project.git cd fashion-mnist-project

2. Instala las dependencias:

pip install -r requirements.txt

## Entrenamiento

Ejecuta el script para entrenar el modelo:

python fashion_mnist_classifier.py --epochs=10

También puedes entrenar el modelo en Vertex AI subiendo el código al bucket de Google Cloud y configurando el entorno.

## Resultados

El modelo alcanzó una precisión del 87.8% en el conjunto de prueba.

