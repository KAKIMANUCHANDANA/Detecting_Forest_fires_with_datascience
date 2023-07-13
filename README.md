import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from skimage import feature
from PIL import Image

# Step 1: Load and preprocess the dataset
data = pd.read_csv('dataset.csv')  # Load the dataset (assuming it's in CSV format)
images = []
labels = []

for index, row in data.iterrows():
    image_path = row['image_path']
    label = row['label']
    image = Image.open(image_path)  # Open the image using PIL
    image = image.resize((64, 64))  # Resize the image to a fixed size
    image = np.array(image)  # Convert the image to a NumPy array
    images.append(image)
    labels.append(label)

X = np.array(images)
y = np.array(labels)

# Step 2: Extract features from the images
features = []
for image in X:
    gray = np.mean(image, axis=2)  # Convert the image to grayscale
    hist = feature.hog(gray)  # Extract Histogram of Oriented Gradients (HOG) features
    features.append(hist)

X = np.array(features)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# Step 5: Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
