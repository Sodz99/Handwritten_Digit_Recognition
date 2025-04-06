import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 to 784
    layers.Dense(128, activation='relu'),  # Hidden layer
    layers.Dropout(0.2),                   # Dropout for regularization
    layers.Dense(10, activation='softmax') # Output layer (10 classes)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
predictions = model.predict(x_test)

# Show example
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {tf.argmax(predictions[0]).numpy()}, Actual: {y_test[0]}")
plt.axis('off')
plt.show()
