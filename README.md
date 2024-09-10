# Image-classification with CNN

`This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images using the CIFAR-10 dataset. The project is implemented using Python with the help of libraries like TensorFlow and OpenCV for image processing.`

## Project Structure

- `run.py`: The main script for training and evaluating the model.
- `utils.py`: Utility functions for displaying sample predictions and plotting the confusion matrix.
- `data_loader.py`: Script for loading and preprocessing the dataset.
- `model.py`: Defines the CNN architecture.
- `train.py`: Contains functions for training the model.

## Requirements

- Python 3.x
- TensorFlow or PyTorch
- OpenCV
- scikit-learn
- seaborn
- matplotlib
  To install these dependencies, run the following command:
  `pip install tensorflow opencv-python matplotlib scikit-learn seaborn`
  `pip install python`

##Dataset
`The project uses the CIFAR-10 dataset, which is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images.
The dataset is automatically downloaded when you run the project.`

##How to Run the Project:
-Clone this repository (or download the source code).
-Install the required dependencies using pip as described above.
-Run the project by executing run.py:
`python run.py`

##The program will:
-Load the CIFAR-10 dataset.
-Preprocess the images.
-Build a Convolutional Neural Network (CNN).
-Train the CNN on the CIFAR-10 dataset.
-Evaluate the model and print the test accuracy.

##Model Overview
'The CNN architecture used in this project is as follows:'
-Input Layer: 32x32x3 (RGB image)
-Convolutional Layers: 3 layers with ReLU activation
-MaxPooling Layers: Pooling layers to reduce the spatial dimensions
-Flatten Layer: Flatten the output from convolutional layers
-Fully Connected Layers: Dense layers to compute the final classification
-Output Layer: Softmax layer with 10 units (one for each class)

##Results
`After training, the model will output its accuracy on the test set. The expected test accuracy should range from 60% to 70% with this simple architecture.`
