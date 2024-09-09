from data_loader import load_data
from model import create_cnn
from train import train_model

x_train, x_test, y_train, y_test = load_data()

model = create_cnn()

train_model(model, x_train, y_train, x_test, y_test)

