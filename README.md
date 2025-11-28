# VPC3_TP_EuroSat_Vit
Integrantes: 

Julian Ferreira

Amilcar Rincon 

Jorge Chavez

Repositorio para el Trabajo Práctico VPC3: clasificación de imágenes satelitales del dataset **EuroSat** utilizando modelos de **Vision Transformer (ViT)**.

## Objetivo del proyecto

Entrenar y evaluar un modelo de clasificación de imágenes que distinga entre distintas clases de uso de suelo (farmland, forest, highway, etc.) usando el dataset EuroSat y un modelo tipo Vision Transformer.

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
