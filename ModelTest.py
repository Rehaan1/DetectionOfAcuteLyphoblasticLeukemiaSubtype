import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the CNN model from the .h5 file
model = load_model('clahe-and-gabor-model2.h5')  # Replace 'your_model_file.h5' with the actual file name

# Your code dictionary
code = {"Benign": 0, "Early": 1, "Pre": 2, "Pro": 3}

def get_code(n):
    for x, y in code.items():
        if n == y:
            return x
        
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

    return enhanced_img

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

def preprocess_and_predict(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Process the image
    processed_img = enhance(img)

    # Resize the image to match the input size expected by the model
    resized_img = cv2.resize(processed_img, (224, 224))  # Adjust the size as per your model's input size

    # Expand the dimensions to match the model's input shape
    input_img = np.expand_dims(resized_img, axis=0)

    # Predict the class
    prediction = model.predict(input_img)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    print("Prediction: ", prediction)
    print("Predicted class index: ", predicted_class_index)

    # Get the class label
    predicted_class = get_code(predicted_class_index)

    return predicted_class

# Example usage:
image_path = 'WBC-Malignant-Pro-297.jpg'  # Replace with the path to your image file
predicted_class = preprocess_and_predict(image_path)
print(f"The predicted class is: {predicted_class}")
