import os
import cv2
import shutil


class SnapShot:
    @staticmethod
    def extract_frames(video_location, output_directory, interval):
        """Takes in a video and saves one screen shot per interval.

        Args:
            video_location (string): the path and video name as a string
            output_directory (string): location to save the images
            interval (float): seconds per image
            return: None
        """
        cap = cv2.VideoCapture(video_location)  # opens the video object
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # save the FPS of the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # save total frame count of video

        # parse video name from path, create subdir for video in output dir
        video_name = os.path.splitext(os.path.basename(video_location))[0]
        video_output_directory = os.path.join(output_directory, video_name)
        os.makedirs(video_output_directory, exist_ok=True)

        # calculate how many frames in current interval
        interval_frames = int(fps * interval)

        for i in range(0, frame_count, interval_frames):
            arr_frame = []
            arr_lap = []
            
            # Read frames for the specified interval
            for j in range(min(interval_frames, frame_count - i)):
                success, frame = cap.read()
                if not success:
                    break
                laplacian = cv2.Laplacian(frame, cv2.CV_64F).var()  # Takes the 'clearest' image in the time range
                arr_lap.append(laplacian)
                arr_frame.append(frame)

            if arr_lap:
                selected_frame = arr_frame[arr_lap.index(max(arr_lap))]
                output_filename = os.path.join(video_output_directory, f"{video_name}_{i+1}.jpg")
                cv2.imwrite(output_filename, selected_frame)
        
        cap.release()


    @staticmethod
    def process_new_dataset(input_dir, output_dir):
        """Iterates through each mp4 in a directory, renames the video,
        and processes it for still images.

        Args:
            input_directory (string): the name of the target directory
            output_dir (string): the name of the destination directory for video stills
            return: None
        """
        parent_dir = os.path.dirname(input_dir)
        output_directory = os.path.join(parent_dir, output_dir)
        os.makedirs(output_directory, exist_ok=True)

        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".mp4"):
                input_path = os.path.join(input_dir, filename)

                # Rename the video file to the desired format (e.g., dog001.mp4)
                video_name = f"dog{len(os.listdir(output_directory)) + 1:03d}"
                renamed_path = os.path.join(input_dir, f"{video_name}.mp4")
                shutil.move(input_path, renamed_path)

                SnapShot.extract_frames(renamed_path, output_directory)


    @staticmethod
    def generate_dataset_from_videos(input_dir, output_dir, pic_interval):
        """Iterates through videos in the input directory and saves images,
        organize by video, taken at the given pic_interval into the output directory.

        Args:
            input_dir (string): directory holding .mp4 files
            output_dir (string): directory for storing new images
            pic_interval (float): seconds per image
        """
        parent_dir = os.path.dirname(input_dir)
        output_directory = os.path.join(parent_dir, output_dir)
        os.makedirs(output_directory, exist_ok=True)

        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".mp4"):
                SnapShot.extract_frames(os.path.join(input_dir, filename),
                                    output_directory,
                                    pic_interval)
