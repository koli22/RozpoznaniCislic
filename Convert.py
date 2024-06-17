import os
import cv2
import numpy as np

def calculate_black_percentage(image):
    total_pixels = image.size
    black_pixels = np.count_nonzero(image == 0)  # Count pixels with value 0 (black)
    return (black_pixels / total_pixels) * 100

def process_images(input_path, output_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate through all files in input_path
    for filename in os.listdir(input_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load image
            img_path = os.path.join(input_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Calculate percentage of black pixels
            black_percentage = calculate_black_percentage(img)

            if black_percentage <= 50.0:
                # Apply threshold to convert to black and white
                _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

                # Resize image to 25x25 pixels
                resized_img = cv2.resize(binary_img, (25, 25))

                # Save processed image
                output_img_path = os.path.join(output_path, filename)
                cv2.imwrite(output_img_path, resized_img)

                print(f"Processed: {filename}")
            else:
                # Delete the image if more than 50% is black
                os.remove(img_path)
                print(f"Deleted: {filename}")

if __name__ == "__main__":
    for i in range(0,10):
        input_folder = "10000/" + str(i) + "/"
        output_folder = str(i) + "o"

        process_images(input_folder, output_folder)
