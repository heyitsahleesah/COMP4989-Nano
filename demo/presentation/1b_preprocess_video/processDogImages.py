import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO
from enum import Enum, auto


class DogStatus(Enum):
    NO_FILE = auto()
    NO_DOG = auto()
    TOO_MANY_DOGS = auto()
    VALID = auto()
    INVALID = auto()


class ProcessDogImages:
    """this class is intended for use on datasets with a directory structure as follows:
    parent_directory/
        subdirectory_containing_images001/
        subdirectory_containing_images002/
        etc.
    """
    def __init__(self, confidence_threshold=0.5, dog_threshold=20, image_count_min=10) -> None:
        self.model = YOLO("yolov8m.pt")  # Load YOLOv8 model
        self.confidence_threshold = confidence_threshold
        self.dog_codes = DogStatus
        self.dog_threshold = dog_threshold  # value is percentage e.g. 20 = 20%
        self.image_count_minimum = image_count_min

    def determine_dog_proportion_is_valid(self, image_path):
        """Check how big the dog is in relation to the photo and determines validity
        based on confidence_threshold and dog_threshold settings.

        Args:
            image_path (string): the file path of the image

        Returns:
            Enum: an enum describing whether the image is valid data or not
        """

        img = cv2.imread(str(image_path))
        
        if len(img) == 0:
            print(f"ERROR: Image at {image_path} not found.")
            return self.dog_codes.NO_DOG
        
        results = self.model(img)

        # Filter results to get only 'dog' class detections above the confidence threshold
        dog_results = [box for box in results[0].boxes if box.cls == 16 and box.conf > self.confidence_threshold]

        if len(dog_results) > 1:
            print(f"Invalid Number of dogs detected in image: {len(dog_results)}")
            return self.dog_codes.TOO_MANY_DOGS
        elif len(dog_results) == 0:
            print(f"No dogs found in image.")
            return self.dog_codes.NO_DOG

        # Calculate the total area of the image
        total_area = img.shape[0] * img.shape[1]

        # Calculate the total area occupied by dog
        # xmax - xmin * ymax - ymin
        print(dog_results[0].xyxy[0])
        dog_area = (dog_results[0].xyxy[0, 2] - dog_results[0].xyxy[0, 0]) * (dog_results[0].xyxy[0, 3] - dog_results[0].xyxy[0, 1])
        # Calculate the percentage of the image occupied by dogs
        percentage_dog_area = (dog_area / total_area) * 100

        print(f"The percentage of {image_path} occupied by dogs is: {percentage_dog_area:.2f}%")
        return self.dog_codes.VALID if percentage_dog_area > self.dog_threshold else self.dog_codes.INVALID
    

    def test_images_in_folder(self, target_directory, invalid_dir=None):
        """Checks the proportion of a dog vs. photo size for each image in a directory. If an image is invalid,
        it is moved to subfolder of the same name as the target_directory in the invalid_data folder.

        Args:
            target_directory (string): the current target directory to check

        """
        target_dir = Path(target_directory)
        if invalid_dir == None:
            invalid_dir = target_dir.parent / 'invalid_data'

        for file in target_dir.iterdir():
            if file.suffix.lower() in ['.jpg', '.jpeg']:
                dog_proportion = self.determine_dog_proportion_is_valid(file)
                print(f"{file} status: {dog_proportion}")

            if dog_proportion != self.dog_codes.VALID:
                invalid_subdirectory = invalid_dir / file.parent.name
                invalid_subdirectory.mkdir(parents=True, exist_ok=True)
                destination_path = invalid_subdirectory / file.name
                shutil.move(file, destination_path)
    

    def sort_dataset(self, data_directory):
        """Traverses data_directory's subfolders and determines if the data (images)
        within are valid (contain single dog and large enough) or not (dog not detected, too small, multiples)

        Args:
            data_directory (string): directory containing subdirectories of dog images to be processed
        """
        data_dir = Path(data_directory)
        invalid_data_dir = data_dir / "invalid_data"
        invalid_data_dir.mkdir(exist_ok=True)
        
        for subdir in data_dir.iterdir():
            if subdir == invalid_data_dir:
                continue
            
            self.test_images_in_folder(subdir)
        
        # loop through and find all ineligible data (fewer than 10 images)
        # move images to invalid data directory - delete empty folder
        for subdir in data_dir.iterdir():
            if subdir.is_dir() and subdir != invalid_data_dir:
                print(f"Checking {subdir}")
                image_count = sum(1 for file in subdir.iterdir() if file.suffix.lower() in ['.jpg', '.jpeg'])            

                # if fewer than image_count_minimum images, it is invalid
                if image_count < self.image_count_minimum:
                    invalid_subdir = invalid_data_dir / subdir.name
                    invalid_subdir.mkdir(parents=True, exist_ok=True)

                    for file in subdir.iterdir():
                        destination_path = invalid_subdir / file.name
                        shutil.move(file, destination_path)
                        print(f"{file} moved to {destination_path}")


    @staticmethod
    def rename_folders_by_count(data_directory):
        """traverses a directory containing subdirectories of individual dogs, 
        and renames the folders dog001, dog002, etc.) and files within 
        consecutively (dog001_1, dog001_2, etc.). Assumes data has been preprocessed,
        so will also delete empty directories.

        Args:
            data_directory (string): directory containing subdirectories of dog images
        """
        target_dir = Path(data_directory)

        for i, subdirectory in enumerate(target_dir.iterdir(), start=1):
            if subdirectory.is_dir():
                # Check if the folder is empty
                if not any(subdirectory.iterdir()):
                    # If empty, delete it
                    subdirectory.rmdir()
                    print(f"Deleted empty folder: {subdirectory}")
                else:
                    # If not empty, rename it and its files
                    new_folder_name = f"dog{i:03d}"
                    new_folder_path = subdirectory.parent / new_folder_name
                    subdirectory.rename(new_folder_path)
                    print(f"Renamed folder: {subdirectory} to {new_folder_path}")

                    # Rename files in the folder
                    for j, file_path in enumerate(new_folder_path.iterdir(), start=1):
                        new_file_name = f"{new_folder_name}_{j:03d}.jpg"
                        new_file_path = new_folder_path / new_file_name
                        file_path.rename(new_file_path)
                        print(f"Renamed file: {file_path} to {new_file_path}")
