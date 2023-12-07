import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO



class Crop:
    def __init__(self):
        self.model = YOLO("yolov8m.pt")  # Load YOLOv8 model


    def get_dog_xy(self, image):
        img = cv2.imread(str(image))
        results = self.model(img)

        # Filter results to get only 'dog' class detections above the confidence threshold
        dog_results = [box for box in results[0].boxes if box.cls == 16]

        try:
            # Get dog coordinates from box
            xmax = dog_results[0].xyxy[0, 2]
            xmin = dog_results[0].xyxy[0, 0]
            ymax = dog_results[0].xyxy[0, 3] 
            ymin = dog_results[0].xyxy[0, 1]
            return (map(int, [xmin, xmax, ymin, ymax]))

        except Exception as er:
            print(er)
            return None
       
    
    def crop_dog_out_of_image(self, image):
        img = Path(image)
        coords = self.get_dog_xy(img)
        if coords is None:
            print(f"No dog detected in {img.name} - skipping")
            return
        
        xmin, xmax, ymin, ymax = coords
        image = cv2.imread(str(img))

        # crop dog out
        cropped_dog = image[ymin:ymax, xmin:xmax]
        
        return cropped_dog


    def process_single_directory(self, input_directory):
        input_dir = Path(input_directory)
        images = [file for file in input_dir.iterdir() if file.suffix in ['.jpg', '.jpeg']]
        for image in images:
            cropped_dog = self.crop_dog_out_of_image(image)
            cv2.imwrite(str(image), cropped_dog)


    def process_directories(self, input_directory, output_directory):
        input_dir = Path(input_directory)
        output_dir = Path(output_directory)

        subdirs = [subdirectory for subdirectory in input_dir.iterdir() if subdirectory.is_dir()]
        
        for subdir in subdirs:
            images = [file for file in Path(subdir).iterdir() if file.suffix in ['.jpg', '.jpeg']]
            for image in images:
                cropped_dog = self.crop_dog_out_of_image(image)
                output_subdir = output_dir / image.parent.name
                output_subdir.mkdir(parents=True, exist_ok=True)

                cropped_dog_path = output_subdir / image.name
                print(f"Saving {image.name} to {cropped_dog_path}")
                cv2.imwrite(str(cropped_dog_path), cropped_dog)


if __name__ == "__main__":
    x = Crop()
    x.process_directories("dog_images", "cropped_dog_images")