import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from os.path import join
#from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Function to load class names from class_names.csv
def load_class_names(file_path):
    with open(file_path, 'rb') as file:
        class_names = pickle.load(file)
    return class_names

# Function to load a single image
def load_single_image(image_path, target_size=(128, 128)):
    print('Loading image...')
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    # test_img = np.expand_dims(test_img, axis=0)
    img = img/255
    img = np.asarray([img])
    return img

# Function to perform predictions and print top class predictions
def predict_and_print_top_classes(model, img_array, class_names):
    print('Running model with image...')
    predictions = model.predict(img_array)
    class_names = {v: k.split('-', 1)[1] for k, v in class_names.items()}
    
    # Get the top k predictions and their indices
    top_prediction_index = predictions.argmax(axis=1)[0]
    
    # Get the class name and probability of the top prediction
    top_class_name = class_names[top_prediction_index]
    top_probability = predictions[0, top_prediction_index]
    
    print("\nTop Prediction:")
    print(f"{'*' * 50}")
    print(f"  🐶  Class: {top_class_name}")
    # print(f"  🔮  Probability: {top_probability:.2%}")
    print(f"{'*' * 50}")

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
device = tf.config.list_physical_devices('GPU')
print(device)
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

print("starting")
# File paths
model_file_path = './model/ResNet50V2_DataAug.hdf5'
checkpoint_path = './checkpoint/ResNet50V2DataAug'
class_names_file_path = './label2index.pickle'
image_path = 'image/sample.jpg'  # Adjust the path accordingly

print("Loading model...")
model = load_model(model_file_path, compile=False)
print('Model loaded.')

print("Compiling model...")
model.compile()
print('model compiled.')

print("Loading weights...")
model.load_weights(checkpoint_path)
print('Weights loaded.')

print("Loading class names...")
class_names = load_class_names(class_names_file_path)
print('Class names loaded.')

target_size = (128, 128)
img_array = load_single_image(image_path, target_size)
print("Image loaded.")

predict_and_print_top_classes(model, img_array, class_names)