import csv
import datetime
import pickle

from tensorflow.python.keras.layers import Conv2D, Dropout
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import constants as constant


class ConvolutionalNeuralNetworkModel:
    def __init__(self, train_data_symbol_count):
        self.__train_data_symbol_count = train_data_symbol_count
        # initialize the CNN
        self.__classifier = Sequential()
        self.__history = None

    def create(self):
        # Step 1: convolution
        self.__classifier.add(Convolution2D(5, 5, input_shape=(50, 50, 1), padding='same', activation='relu'))

        # Step 2: pooling
        self.__classifier.add(MaxPooling2D(pool_size=(4, 4)))

        # Add a convolutional layer
        self.__classifier.add(Convolution2D(15, 5, input_shape=(50, 50, 1), padding='same', activation='relu'))

        # Add another max pooling layer
        self.__classifier.add(MaxPooling2D(pool_size=(4, 4)))

        # Step 3: Flattening
        self.__classifier.add(Flatten())

        # Step 4: Full connection
        self.__classifier.add(Dense(units=self.__train_data_symbol_count, activation='softmax'))

        # Compiling the CNN
        self.__classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def create_another_model(self):
        # Step 1: convolution
        self.__classifier.add(Convolution2D(5, 5, input_shape=(50, 50, 1), padding='same', activation='relu'))

        # Step 2: pooling
        self.__classifier.add(MaxPooling2D(pool_size=(4, 4)))

        # Add a convolutional layer
        self.__classifier.add(Convolution2D(15, 5, input_shape=(50, 50, 1), padding='same', activation='relu'))

        # Add another max pooling layer
        self.__classifier.add(MaxPooling2D(pool_size=(4, 4)))

        # Step 3: Flattening
        self.__classifier.add(Flatten())

        # Step 4: Full connection
        self.__classifier.add(Dense(units=self.__train_data_symbol_count, activation='relu'))
        self.__classifier.add(Dropout(0.25))

        self.__classifier.add(Dense(units=self.__train_data_symbol_count, activation='softmax'))

        # Compiling the CNN
        self.__classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def create_another_model_99_accuracy(self):
        self.__classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
        self.__classifier.add(MaxPooling2D(pool_size=(2, 2)))

        self.__classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.__classifier.add(MaxPooling2D(pool_size=(2, 2)))

        self.__classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.__classifier.add(MaxPooling2D(pool_size=(2, 2)))

        self.__classifier.add(Flatten())
        self.__classifier.add(Dense(128, activation='relu'))
        self.__classifier.add(Dropout(0.20))
        self.__classifier.add(Dense(units=self.__train_data_symbol_count, activation='softmax'))

        self.__classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.__classifier.summary()

    def train(self):
        print('Fitting the CNN to the images')
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        print('Train set loaded')
        train_set = train_datagen.flow_from_directory(constant.TRAIN_DATA_DIRECTORY,
                                                      target_size=(50, 50),
                                                      color_mode='grayscale',
                                                      batch_size=32,
                                                      class_mode='categorical')

        print('Validation set loaded')
        validation_set = train_datagen.flow_from_directory(constant.VALIDATION_DATA_DIRECTORY,
                                                           target_size=(50, 50),
                                                           color_mode='grayscale',
                                                           batch_size=32,
                                                           class_mode='categorical')

        print('Training the network')
        self.__history = self.__classifier.fit_generator(
            train_set,
            steps_per_epoch=500,
            epochs=100,
            validation_data=validation_set,
            validation_steps=10
        )

    def save_metrics_and_model(self):
        now = datetime.datetime.now()
        date_index = now.strftime("%Y-%m-%d")

        print('Saving the history metrics in a file;')
        w = csv.writer(open(constant.MODEL_METRICS_DIRECTORY
                            + "history_metrics_" + date_index + ".csv", "w"))
        for key, val in self.__history.history.items():
            w.writerow([key, val])

        with open(constant.MODEL_METRICS_DIRECTORY + 'train_history_dict_' + date_index + '.txt', 'wb') as file_pi:
            pickle.dump(self.__history.history, file_pi)

        print('Saving the model')
        self.__classifier.save(constant.MODEL_METRICS_DIRECTORY + 'model_saved_' + date_index + '.h5')


model = ConvolutionalNeuralNetworkModel(30)
model.create()
# model.create_another_model_99()
model.summary()
model.train()
model.save_metrics_and_model()
