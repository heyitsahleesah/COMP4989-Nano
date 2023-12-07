# This script demonstrates the proper usage for taking a video file of a missing
# dog as input, processing it, and storing the cropped images in the appropriate
# destination directory
import os
import random
import shutil
from crop import Crop
from processDogImages import ProcessDogImages
from pathlib import Path


# ensure that both directories exist prior to running pipeline
input_directory = Path("dog_images")
invalid_directory = Path("invalid_dog_images")

# get path to video file in source_directory
# Note: this directory should hold a single video file at a time - no other files permitted.
# Note: Video file must be .mp4
target_file = list(input_directory.glob('*.jpg'))
target_file_path = str(target_file[0].resolve())

# create ProcessDogImages object to check if dog is sufficient proportion of images
# move invalid images to invalid_directory for review/deletion
dog_processor = ProcessDogImages()
dog_processor.test_images_in_folder(input_directory, invalid_directory)

# Crop valid images using bounding box of dog
cropper = Crop()
cropper.process_single_directory(input_directory)

# Move one of the results to the ../generic_predict/image/ and ../modified_predict/image/ folders
demo_1_path = "../generic_predict/image"
demo_2_path = "../modified_predict/image"

image_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
selected_image = random.choice(image_files)

source_path = os.path.join(input_directory, selected_image)
demo1_path = os.path.join(demo_1_path, 'sample.jpg')
demo2_path = os.path.join(demo_2_path, 'sample.jpg')

shutil.copy(source_path, demo1_path)
shutil.copy(source_path, demo2_path)