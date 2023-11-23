import os
import cv2
import numpy as np
from skimage import exposure 
from skimage.exposure import match_histograms 

def enhance(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gabor filter to the grayscale image
    gabor_filtered = apply_gabor_filter(gray)

    # Apply CLAHE to the Gabor-filtered image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gabor_filtered)

    # Merge the enhanced channel with the original BGR channels
    enhanced_img = cv2.merge([cl, img[:, :, 1], img[:, :, 2]])

    # Apply Laplacian filter for edge emphasis directly on the enhanced image
    laplacian = cv2.Laplacian(enhanced_img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # Combine the enhanced image and Laplacian-filtered image
    enhanced_with_laplacian = cv2.addWeighted(enhanced_img, 0.7, laplacian, 0.3, 0)

    return enhanced_with_laplacian

def apply_gabor_filter(img):
    # Define Gabor filter parameters
    ksize = 9
    sigma = 2.0
    theta = np.pi / 4.0
    lambd = 10.0
    gamma = 0.5

    # Create Gabor filter
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)

    # Apply Gabor filter to the grayscale image
    gabor_filtered = cv2.filter2D(img, cv2.CV_8UC1, gabor_kernel)

    return gabor_filtered

input_directory = './Original/Pro/'
output_directory = './Preprocessed/Pro/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# List all files in the input directory
input_files = os.listdir(input_directory)

for file in input_files:
    if file.endswith('.jpg') or file.endswith('.png'):
        # Read the image
        img = cv2.imread(os.path.join(input_directory, file))
        
        # Process the image
        processed_img = enhance(img)
        
        # Save the processed image to the output directory
        output_file = os.path.join(output_directory, file)
        cv2.imwrite(output_file, processed_img)
        print(f"Processed and saved {file} to {output_file}")
