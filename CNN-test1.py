import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from split_images import split_data
import joblib

# uncomment if needing to split data
# get the images and split them
# input_folder = "DFN_dataset/after_4_bis"
# output_folder = "Data_split"
# split_ratio = 0.8  # 80% for training, 20% for testing
# min_images = 5  # Minimum number of images required

# split_data(input_folder, output_folder, split_ratio, min_images)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=90,
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load the training dataset
train_generator = train_datagen.flow_from_directory(
    "Data_split/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",  # Update to categorical for multi-class classification
)

# Get the class labels
class_labels = list(train_generator.class_indices.keys())

# Specify the file path where you want to save the class labels
output_file_path = "class_labels.txt"

# Open the file in write mode
with open(output_file_path, "w") as file:
    # Write each class label to the file
    for label in class_labels:
        file.write(label + "\n")

# Inform the user that the class labels have been saved
print(f"Class labels saved to {output_file_path}")

# Load the testing dataset
test_generator = test_datagen.flow_from_directory(
    "Data_split/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",  # Update to categorical for multi-class classification
)

# Build the CNN model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(512, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.5))

# Update the number of neurons in the output layer to match the number of classes
num_classes = len(train_generator.class_indices)
model.add(layers.Dense(num_classes, activation="sigmoid"))

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",  # Update to categorical_crossentropy for multi-class classification
    metrics=["accuracy"],  # Update to accuracy for classification
)

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the model
test_loss, test_mae = model.evaluate(test_generator)
print(f"Test Mean Absolute Error: {test_mae}")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")

# Specify the file path where you want to save the output
output_file_path = "output.txt"

# Make predictions
predictions = model.predict(test_generator)

# Get the actual labels from the test generator
actual_labels = test_generator.classes

# Set the number of samples to print
num_samples_to_print = 25

# Open the file in write mode
with open(output_file_path, "w") as file:
    # Print the actual labels and predicted values for a subset of samples
    for i, (actual, predicted) in enumerate(zip(actual_labels, predictions)):
        # Construct the string to be written to the file
        output_str = f"Sample {i+1}: Actual: {actual}, Predicted: {predicted[0]}\n"

        # Print to the console
        print(output_str, end="")

        # Write the string to the file
        file.write(output_str)

        # Break the loop after printing the desired number of samples
        if i == num_samples_to_print - 1:
            break

# Inform the user that the output has been saved
print(f"Output saved to {output_file_path}")

# Save the model
model.save("CNN_v1.h5")
