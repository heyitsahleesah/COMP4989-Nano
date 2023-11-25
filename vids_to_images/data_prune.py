# a short script for deleting non-dog images in the Can Humans Fly? dataset.
import csv
import os


def delete_video(file_name):
    """Deletes video matching file_name and .mp4 file type.

    Args:
        file_name (string): an existing file in the working directory
        return: None
    """
    video_path = os.path.join("clips320H", file_name + ".mp4")
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Deleted video: {video_path}")
    else:
        print(f"Video not found: {video_path}")


def process_csv(csv_file_path, allowed_values):
    """Reads in all rows of CSV file - if a row is not in a permitted class, delete video
    associated with it.

    Args:
        csv_file_path (string): the videoset csv file for this dataset
        allowed_values (list): list of permitted file classes
    """
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            file_name = row[0]
            value = int(row[1])

            if value not in allowed_values:
                delete_video(file_name)


# Replace 'your_csv_file.csv' with the actual path to your CSV file
csv_file_path = 'videoset.csv'

# Replace [list_of_allowed_values] with the list of integers that are allowed
allowed_values = [71, 72, 73, 74, 75, 76, 77, 78, 79]

process_csv(csv_file_path, allowed_values)
