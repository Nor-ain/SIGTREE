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
    resized = cv2.resize(img, (200, 200))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    # Texture features
    sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=5)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    gradient_magnitude = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    edge = cv2.Canny(resized, 100, 200)
    # Shape features
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / float(h)
    else:
        aspect_ratio = 0
    # Color features
    hist = np.concatenate([cv2.calcHist([hsv], [i], None, [256], [0, 256]).flatten() for i in range(3)])

    # Combine all features
    feature_vector = np.concatenate((hist, gradient_magnitude.flatten(), [aspect_ratio]))
    return feature_vector


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
            'accuracy': f'{accuracy * 100:.2f}%' if accuracy != 'N/A' else accuracy
        }
        return render(request, 'output_image.html', context)

    return render(request, 'predict_image.html')
