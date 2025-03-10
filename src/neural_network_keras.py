import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.datasets import mnist

def cargar_datos():
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()
    return train_data_x, train_labels_y, test_data_x, test_labels_y

def normalizar_datos(train_data_x, test_data_x, train_labels_y, test_labels_y):
    x_train = train_data_x.reshape(60000, 28*28)
    x_train = x_train.astype('float32')/255
    y_train = to_categorical(train_labels_y)
    
    x_test = test_data_x.reshape(10000, 28*28)
    x_test = x_test.astype('float32')/255
    y_test = to_categorical(test_labels_y)
    
    return x_train, y_train, x_test, y_test

def crear_modelo():
    model = Sequential([
        Input(shape=(28*28,)),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    return model

def entrenar_modelo(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=8, batch_size=128)
    model.evaluate(x_test, y_test)

def ejecutar_red_neuronal():
    # Cargar datos
    train_data_x, train_labels_y, test_data_x, test_labels_y = cargar_datos()
    
    # Mostrar información sobre los datos
    print(train_data_x.shape)
    print(train_labels_y[0])
    
    # Normalizar datos
    x_train, y_train, x_test, y_test = normalizar_datos(train_data_x, test_data_x, train_labels_y, test_labels_y)
    
    # Crear modelo
    model = crear_modelo()
    model.summary()
    
    # Entrenar modelo
    entrenar_modelo(model, x_train, y_train, x_test, y_test)


