import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from data_loader import load_data
from model import create_cnn
from train import train_model
from utils import show_sample_predictions, plot_confusion_matrix

x_train, x_test, y_train, y_test = load_data()

model = create_cnn()

history = train_model(model, x_train, y_train, x_test, y_test)

predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

show_sample_predictions(x_test, y_test, predicted_classes, class_names)

plot_confusion_matrix(y_test, predicted_classes, class_names)


