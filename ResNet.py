import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.layers import Add
import keras

# get data
cifar10 = tf.keras.datasets.cifar10.load_data()

X_train, y_train = cifar10[0][0], cifar10[0][1]
X_test, y_test = cifar10[1][0], cifar10[1][1]

img_rows, img_cols = 32, 32
pixelValuesRange = 255
num_classes = 10

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= pixelValuesRange
X_test /= pixelValuesRange

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# ResNet (one block of three layers)
epochs = 3
batch_size = 32

shape = X_train.shape[1:]
inputs = Input(shape=shape)

x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(inputs)
x1 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(x)
x2 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(x1)
added = Add()([x, x2])
added = Flatten()(added)
predictions = Dense(num_classes, activation='softmax')(added)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='testing accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='testing loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

print("end")
