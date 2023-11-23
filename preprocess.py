import os
import cv2
import numpy as np
from skimage import exposure 
from skimage.exposure import match_histograms 

def segmentWBC(img, referenceImage):
    
    img = match_histograms(img, referenceImage, multichannel=True)
    
    # Apply a bilateral filter for noise reduction while preserving edges
    # The d parameter controls the diameter of each pixel neighborhood for filtering, 
    # while sigmaColor and sigmaSpace control the color and spatial Gaussian standard deviations, respectively. 

    #img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    
    # Convert the RGB image to L*a*b* color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Split the L*a*b* image into L, a, and b channels
    H, S, V = cv2.split(img_hsv)
    
    # Apply a threshold to the image to separate the WBCs from the background
    threshold = 120
    wbcs_mask = cv2.threshold(S, threshold, 255, cv2.THRESH_BINARY)[1]

    # Combine the actual image and the WBCs mask image to create the final image
    final_img = cv2.bitwise_and(img, img, mask=wbcs_mask)
    
    return final_img

input_directory = './Original/Pro/'
output_directory = './Preprocessed/Pro/'

refImg = cv2.imread('WBC-Benign-001.jpg')

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# List all files in the input directory
input_files = os.listdir(input_directory)

for file in input_files:
    if file.endswith('.jpg') or file.endswith('.png'):
        # Read the image
        img = cv2.imread(os.path.join(input_directory, file))
        
        # Process the image using segmentWBC
        processed_img = segmentWBC(img, refImg)
        
        # Save the processed image to the output directory
        output_file = os.path.join(output_directory, file)
        cv2.imwrite(output_file, processed_img)
        print(f"Processed and saved {file} to {output_file}")