# VPC3_TP_EuroSat_Vit
Integrantes: 

Julian Ferreira

Amilcar Rincon 

Jorge Chavez

Repositorio para el Trabajo Práctico VPC3: clasificación de imágenes satelitales del dataset **EuroSat** utilizando modelos de **Vision Transformer (ViT)**.

## Objetivo del proyecto

Entrenar y evaluar un modelo de clasificación de imágenes que distinga entre distintas clases de uso de suelo (farmland, forest, highway, etc.) usando el dataset EuroSat y un modelo tipo Vision Transformer.


## 📋 Requisitos Previos

Para ejecutar este proyecto necesitas tener instalado:

- **Python 3.12** (Versión probada y recomendada).

## 🚀 Instalación

1. Clona este repositorio (si aplica).
2. Asegúrate de estar en la raíz del proyecto.
3. Instala las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

## 🗂️ Preparación del Dataset
Antes de entrenar los modelos, es necesario descargar y preparar los datos. Ejecuta los siguientes scripts en orden desde la carpeta src/data:

1. Descargar el dataset:

```bash
python src/data/download_dataset.py
```

2. Generar el dataset en formato YOLO:

```bash
python src/data/make_YOLO_dataset.py
```

Nota: Asegúrate de que los scripts se ejecuten correctamente antes de pasar a la siguiente etapa.

## 🧠 Entrenamiento y Evaluación de Modelos

El proyecto cuenta con directorios específicos para diferentes modelos. Para trabajar con uno de ellos:

1. Navega al directorio del modelo deseado (ejemplo: cd src/models/nombre_del_modelo).

2. Ejecuta primero el script de entrenamiento:

```bash 
python train_model.py
```
3. Una vez finalizado el paso anterior, ejecuta el script de inferencia:

```bash 
python predict_model.py
```

## Estructura del proyecto

Este proyecto sigue una estructura estilo *cookiecutter data science*:

```bash
VPC3_TP_EuroSat_Vit/
├── data/
│   ├── raw/         # Datos originales (EuroSat sin modificar)
│   ├── interim/     # Datos transformados parcialmente
│   ├── processed/   # Datos listos para modelar (splits train/val/test)
│   └── external/    # Otros datasets o capas externas (opcional)
├── notebooks/       # Jupyter notebooks de exploración y experimentos
├── models/          # Modelos entrenados (.pth, .pt) y resultados
├── reports/
│   ├── figures/     # Gráficas y visualizaciones generadas
│   └── ...          # Informes en PDF/HTML, etc.
├── src/
│   ├── data/        # Scripts para manejo/preparación de datos
│   ├── features/    # Scripts para construcción de features/dataloaders
│   ├── models/      # Scripts de entrenamiento y predicción
│   └── visualization/ # Scripts para graficar resultados
