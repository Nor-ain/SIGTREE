import base64
import cv2
import joblib
import numpy as np
from django.shortcuts import render

# Load the models
classifier_model = joblib.load('static/classifier_model.joblib')
outlier_detection_model = joblib.load('static/outlier_detection_model.joblib')

# List of class names your model can predict
CATEGORIES = ['MILD', 'SEVERE']


def preprocess(img):
    # Resize the image
    resized = cv2.resize(img, (200, 200))

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)  # Corrected from RGB to HSV to BGR to HSV

    # Calculate the histogram for each channel of the input image - FEATURE EXTRACTION
    hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_saturation = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv], [2], None, [256], [0, 256])

    # Concatenate the histograms
    hist = np.concatenate((hist_hue, hist_saturation, hist_value))

    # Return to flatten histogram into a 1D array
    return hist.flatten()


def predict_image(request):
    # Check if the request method is POST
    if request.method == 'POST':
        # Get the image file from the request
        image_file = request.FILES.get('image')

        if image_file:
            # Convert the image file into a NumPy array for processing
            img_array = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Preprocess the image using the preprocess function
            preprocessed_img = preprocess(img)

            # Outlier detection
            is_outlier = outlier_detection_model.predict([preprocessed_img])[0]
            if is_outlier == -1:
                # Handle cases where the image is considered an outlier
                predicted_label = 'No matching images found. Try one more time.'
                accuracy = 'N/A'
                image_data = 'static/no_image_found.png'
            else:
                # Predict the class
                probabilities = classifier_model.predict_proba([preprocessed_img])[0]
                predicted_label_index = np.argmax(probabilities)
                predicted_label = 'MILD' if predicted_label_index == 0 else 'SEVERE'
                accuracy = probabilities[predicted_label_index]

                # Convert the image to Base64 for displaying on the web
                _, buffer = cv2.imencode('.jpg', img)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                image_data = f'data:image/jpeg;base64,{encoded_image}'
        else:
            image_data = 'No image provided'
            predicted_label = 'No prediction'
            accuracy = 'N/A'

        # Prepare the response
        context = {
            'image_data': image_data,
            'predicted_label': predicted_label,
            'accuracy': f'{accuracy*100:.2f}%' if accuracy != 'N/A' else accuracy
        }
        return render(request, 'output_image.html', context)

    return render(request, 'predict_image.html')
