import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

# Load and preprocess the CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize the images
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train, y_test = to_categorical(y_train, 100), to_categorical(y_test, 100)

# Define the CNN model with L2 regularization
model = models.Sequential()

# Conv Layer 1
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
                        kernel_regularizer=regularizers.l2(0.1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Conv Layer 2
model.add(layers.Conv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularizers.l2(0.1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Conv Layer 3
model.add(layers.Conv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularizers.l2(0.1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer
model.add(layers.Flatten())

# Fully Connected Layer 1
model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

# Output layer
model.add(layers.Dense(100, activation='softmax'))

# Compile the model with a lower initial learning rate
initial_learning_rate = 0.0005
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=64, 
                    validation_data=(x_test, y_test),
                    callbacks=[reduce_lr])

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")