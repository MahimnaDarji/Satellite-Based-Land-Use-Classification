# Satellite Based Land Use Classification

## Project Overview
This project is a satellite based land use classification system built using ResNet18 with transfer learning.

It classifies satellite image patches into four categories:
- Agriculture
- Forest
- Urban
- Water

Along with classification, the project also generates clear visual outputs like Grad-CAM, prediction samples, misclassification samples, a confusion matrix, and a tile based overlay map. The final dashboard combines all outputs in one view.

## Project Highlights
- 4 class land use classification (Agriculture, Forest, Urban, Water)
- ResNet18 with transfer learning
- 96% test accuracy on unseen test data
- Confusion matrix for class wise performance check
- Prediction gallery for visual verification
- Misclassification gallery to study errors
- Grad-CAM visualizations to show where the model focuses
- Tile based overlay map for larger land use view
- Final dashboard that combines all outputs

## Problem Statement
Satellite images show different land patterns like fields, forests, water bodies, and built areas. The goal is to classify these land types correctly and also show clear evidence of model behaviour using visual outputs.

This project focuses on:
- Patch level land use classification
- Class wise performance checking
- Error analysis using wrong prediction samples
- Model focus checking using Grad-CAM
- Converting patch predictions into a larger land use overlay view

## Dataset Description
The dataset contains multiple land cover folders such as:
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

These raw folders are mapped into four final classes:
- Agriculture
- Forest
- Urban
- Water

The dataset is split into:
- Train
- Validation
- Test

This ensures the final accuracy is measured on unseen images.

## Model Details
Model used: **ResNet18**

Training setup:
- Transfer learning (pretrained weights)
- Input size: 224 x 224
- Loss: Cross entropy
- Optimizer: Adam

Final result:
- **96% test accuracy** on unseen test data

## Evaluation Outputs

### Confusion Matrix
Shows class wise predictions and where the model gets confused. Most values are on the diagonal, which means most predictions are correct.

### Prediction Gallery
Shows sample test images with:
- True label
- Predicted label
- Confidence

This helps to visually verify if predictions look correct.

### Misclassification Gallery
Shows only wrong predictions. This helps identify the type of images where the model struggles.

### Grad-CAM Visualization
Grad-CAM highlights the exact areas of the image the model uses while making a prediction. This makes it easier to check whether the model is focusing on useful regions instead of random background.

### Tile Based Land Use Overlay Map
A larger satellite image is divided into tiles, each tile is classified, and then the outputs are combined into a single overlay map. This converts patch predictions into a larger land use view.

## Final Dashboard
The final dashboard combines:
- Confusion matrix
- Prediction gallery
- Misclassification gallery
- Grad-CAM results
- Land use overlay map

This gives a complete view of accuracy, errors, and model behaviour in one image.

## GitHub Repository
https://github.com/MahimnaDarji/Satellite-Based-Land-Use-Classification.git
