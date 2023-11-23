import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the pre-trained model from the H5 file
model = load_model("CNN_v1.h5")

# Load a single test image
test_img_path = "test.jpeg"
test_img = image.load_img(test_img_path, target_size=(224, 224))

# Convert the test image to a numpy array
test_img_array = image.img_to_array(test_img)

# Expand the dimensions to match the model input shape
test_img_array = np.expand_dims(test_img_array, axis=0)

# Preprocess the test input image
test_img_array = preprocess_input(test_img_array)

# Make a prediction
prediction = model.predict(test_img_array)

# Get the predicted class label
predicted_class = np.argmax(prediction)

# Get the class labels from the training data
class_labels_file = "class_labels.txt"
with open(class_labels_file, "r") as file:
    class_labels = [line.strip() for line in file.readlines()]

# Print the test image
plt.subplot(1, 2, 1)
plt.imshow(test_img)
plt.title("Test Image")

# Get the path to the predicted class folder
predicted_class_folder = os.path.join("Data_split/train", class_labels[predicted_class])

# List the images in the predicted class folder
predicted_class_images = os.listdir(predicted_class_folder)

# Display the first image in the predicted class
if predicted_class_images:
    # Construct the path to the first image in the predicted class
    predicted_class_image_path = os.path.join(
        predicted_class_folder, predicted_class_images[0]
    )

    # Load and display the image
    predicted_class_image = image.load_img(predicted_class_image_path)
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_class_image)
    plt.title(f"Closest Match: {class_labels[predicted_class]}")

    plt.show()
else:
    print(f"No images found in the predicted class folder: {predicted_class_folder}")
