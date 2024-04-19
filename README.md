# COMP6721-GroupB: Leaf Disease Classification

## Link to the GitHub Page of the Project: 
https://github.com/MariyaBosy/COMP6721-GroupB.git

## Link to the Video: 

## Overview

This project focuses on developing and training deep learning models for plant and leaf disease classification using Convolutional Neural Networks (CNNs) like MobileNet, ResNet, and VGG. These models are trained on various datasets to identify and classify diseases in plants and leaves, contributing significantly to agricultural health practices.

### Files in the Project

- `MobileNet_Mendeley.ipynb`: MobileNet model trained on the Mendeley dataset.
- `MobileNet_PlantVillage.ipynb`: MobileNet model trained on the PlantVillage dataset.
- `MobileNet_Rice_Leaf.ipynb`: MobileNet model trained on the Rice Leaf dataset.
- `ResNet_CCMT.ipynb`: ResNet model trained on a custom dataset.
- `ResNet_PlantVillage.ipynb`: ResNet model trained on the PlantVillage dataset.
- `ResNet_Rice_Leaf.ipynb`: ResNet model trained on the Rice Leaf dataset.
- `VGG_Mendeley.ipynb`: VGG model trained on the Mendeley dataset.
- `VGG_PlantVillage.ipynb`: VGG model trained on the PlantVillage dataset.
- `VGG_Rice_Leaf.ipynb`: VGG model trained on the Rice Leaf dataset.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Instructions for Training and Validation

1. **Download the Project**: Clone or download the zip file containing all the notebook files.

2. **Set Up Your Environment**:
   - Ensure Python 3.x is installed on your system.
   - Install PyTorch and other required libraries using `pip`:
     ```
     pip install torch torchvision numpy matplotlib pillow
     ```

3. **Obtain the Datasets**:
   - Download the datasets from the provided links :
        PlantVillage Dataset: https://github.com/spMohanty/PlantVillage-Dataset
        Mendeley Dataset 1 (Crop Pest and Disease Detection): https://data.mendeley.com/datasets/bwh3zbpkpv/1
        Mendeley Dataset 2 (Rice Leaf Disease Detection): https://data.mendeley.com/datasets/fwcj7stb8r/1

   - Ensure the datasets are structured appropriately as expected by the notebook scripts.

4. **Train the Models**:
   - Open the desired Jupyter notebook (e.g., `MobileNet_Mendeley.ipynb`) in an environment that supports like Google Collab or Kaggle. Here, we have used Google Collab and Visual Studio Code to train our models.
   - Run the cells sequentially to train the model and observe the training process and outcomes.

5. **Validate and Test the Models**:
   - The notebooks contain sections for validating and testing the models on the test datasets.
   - Follow the instructions within each notebook to perform validation and testing.

## Running Pre-trained Models

- To run a pre-trained model, load the model weights in the notebook and run the prediction cells.
- Ensure the test dataset is correctly loaded and pre-processed as per the model requirements.


## How to obtain the Datasets from an available download link

To obtain the datasets, follow these steps:

1. **PlantVillage Dataset**:
   - Go to the GitHub repository link: (https://github.com/spMohanty/PlantVillage-Dataset).
   - On the repository page, find the 'Code' button and click on it.
   - Select 'Download ZIP' from the dropdown menu to download the dataset.

2. **Mendeley Dataset 1 (Crop Pest and Disease Detection)**:
   - Navigate to the Mendeley Data page: (https://data.mendeley.com/datasets/bwh3zbpkpv/1).
   - Click on the 'Download all' button to download the dataset.

3. **Mendeley Dataset 2 (Rice Leaf Disease Detection)**:
   - Visit the Mendeley Data page for the dataset: (https://data.mendeley.com/datasets/fwcj7stb8r/1).
   - Click on the 'Download all' button to download the dataset.

After downloading, extract the ZIP files and follow the specific instructions in the provided notebooks to use these datasets for training and evaluating the models.


By following these instructions, users can train, validate, and run pre-trained models for plant and leaf disease classification across different datasets using CNN architectures like MobileNet, ResNet, and VGG.