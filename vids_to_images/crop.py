import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO



class Crop:
    def __init__(self, face_cropping=True):
        self.model = YOLO("yolov8m.pt")  # Load YOLOv8 model
        self.face_crop = face_cropping

    
    def get_dog_xy(self, image):
        img = cv2.imread(str(image))
        results = self.model(img)

        # Filter results to get only 'dog' class detections above the confidence threshold
        dog_results = [box for box in results[0].boxes if box.cls == 16 and box.conf > self.confidence_threshold]

        if len(dog_results) > 1:
            print(f"Invalid Number of dogs detected in image: {len(dog_results)}")
            # SORT INTO TOO MANY DOGS FOLDER
            return self.dog_codes.TOO_MANY_DOGS
        elif len(dog_results) == 0:
            print(f"No dogs found in image.")
            # SORT INTO NO DOGS FOLDER
            return self.dog_codes.NO_DOG

        # Calculate the total area of the image
        total_area = img.shape[0] * img.shape[1]

        # Calculate the total area occupied by dog
        # xmax - xmin * ymax - ymin
        print(dog_results[0].xyxy[0])

    
    def crop_dog_out_of_image(self, image):
        pass


    def crop_dog_face_out_of_image(self, image):
        pass

    
    def process_directories(self, input_directory):
        pass

    