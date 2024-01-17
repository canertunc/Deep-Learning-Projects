import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

for_train = ImageDataGenerator(rescale=1.0/255,shear_range=0.3,zoom_range=0.3,horizontal_flip=True)
train_data = for_train.flow_from_directory("data_sets/train",target_size=(64,64),batch_size=32,class_mode="binary")

for_valid = ImageDataGenerator(rescale=1.0/255)
valid_data = for_valid.flow_from_directory("data_sets/valid",target_size=(64,64),batch_size=32,class_mode="binary")

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters = 32,kernel_size=3,activation="relu",input_shape = [64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters = 32,kernel_size=3,activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128,activation="relu"))
cnn.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

cnn.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
cnn.fit(x = train_data,validation_data=valid_data,epochs=50)






