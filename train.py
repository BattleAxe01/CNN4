# import keras op
import tensorflow as tf

# GPU config
gpus = tf.config.list_physical_devices('GPU')
try:
    tf.config.set_soft_device_placement(True)
    # tf.debugging.set_log_device_placement(True)
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
    # tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

except RuntimeError as e:
    print(e)

# inicialize the CNN
from tensorflow.keras import Sequential

model = Sequential()

# inicialize variables
batch = 32
input_size = (128, 128)

# image pre processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory('dataset/train_set', target_size=input_size, batch_size=batch,
                                              class_mode='categorical', shuffle=True)

test_gen = test_datagen.flow_from_directory('dataset/test_set', target_size=input_size, batch_size=batch,
                                            class_mode='categorical', shuffle=True)

# add layers
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model.add(Convolution2D(32, (3, 3), input_shape=(*input_size, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.summary()

# compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# train CNN
model.fit(train_gen, steps_per_epoch=8000 / batch, epochs=10, validation_data=test_gen, validation_steps=2000 / batch)

# save model
model.save("./model")
