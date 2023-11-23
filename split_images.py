import os
import shutil
import random


def split_data(input_folder, output_folder, split_ratio=0.8, min_images=5):
    # Create output folders for training and testing
    train_folder = os.path.join(output_folder, "train")
    test_folder = os.path.join(output_folder, "test")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Iterate through each subfolder
    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        if os.path.isdir(class_path):
            # Check the number of images in the current class folder
            num_images = len(
                [
                    f
                    for f in os.listdir(class_path)
                    if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")
                ]
            )

            # Skip folders with less than 5 images
            if num_images < min_images:
                print(f"Skipping {class_folder} - not enough images.")
                continue

            # Create subfolders in train and test folders
            train_class_path = os.path.join(train_folder, class_folder)
            test_class_path = os.path.join(test_folder, class_folder)
            os.makedirs(train_class_path, exist_ok=True)
            os.makedirs(test_class_path, exist_ok=True)

            # List all files in the current class folder
            files = os.listdir(class_path)
            # Randomly shuffle the files
            random.shuffle(files)

            # Calculate the split index
            split_index = int(split_ratio * len(files))

            # Copy files to the train folder
            for file in files[:split_index]:
                source_path = os.path.join(class_path, file)
                dest_path = os.path.join(train_class_path, file)
                shutil.copy(source_path, dest_path)

            # Copy files to the test folder
            for file in files[split_index:]:
                source_path = os.path.join(class_path, file)
                dest_path = os.path.join(test_class_path, file)
                shutil.copy(source_path, dest_path)
