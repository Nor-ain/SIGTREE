import base64
import cv2
import joblib
import numpy as np
from django.shortcuts import render

# Load the models
scaler = joblib.load('static/scaler.joblib')
decision_tree = joblib.load('static/decision_tree.joblib')
iso_forest = joblib.load('static/iso_forest.joblib')


# Function to preprocess image
def preprocess(img):
    # Resize the image and convert color to hsv
    resized = cv2.resize(img, (128, 128))
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
    aspect_ratio = max([cv2.boundingRect(contour)[2] / float(cv2.boundingRect(contour)[3]) for contour in contours],
                       default=0)

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

            # Preprocess the image
            preprocessed_img = preprocess(img)

            # Perform feature scaling
            single_scaled_features = scaler.transform([preprocessed_img])

            # Predict if the test image is an outlier
            is_outlier = iso_forest.predict(single_scaled_features)[0]
            if is_outlier == -1:
                # Handle cases where the image is considered an outlier
                predicted_label = 'No matching images found. Try one more time.'
                accuracy = 'N/A'
                image_data = 'static/no_image_found.png'
            else:
                # Perform prediction for the image
                probabilities = decision_tree.predict_proba(single_scaled_features)[0]
                predicted_label_index = np.argmax(probabilities)
                categories = ['MILD', 'SEVERE']
                predicted_label = categories[predicted_label_index]
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
