import cv2

class DogFaceDetector:
    def __init__(self, cascade_path):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    

    def detect_dog_face(self, img_path):
        # read image with cv2
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # attempt to detect faces
        dog_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # get bounding box for deteced face:
        for (x, y, w, h) in dog_faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imshow("Dog Face Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cascade_path = "dog_face.xml" 
    image_path1 = "cropped_dog_images/dog023/dog023_009.jpg"
    image_path2 = "cropped_dog_images/dog036/dog036_011.jpg"

    detector = DogFaceDetector(cascade_path)
    detector.detect_dog_face(image_path1)
    detector.detect_dog_face(image_path2)