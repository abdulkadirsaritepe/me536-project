## ME536 Design of Intelligent Machines - Term Project
## Name: Abdulkadir Sarıtepe
## Student ID: 2378669
## Date: 2025-01-30
## Description: This is the main file for the term project of ME536 Design of Intelligent Machines course. The main file includes the GUI for the project. The GUI includes the following functionalities:
## 1. Select the project directory
## 2. Select the mode (Training, Validation, Testing, Live)
## 3. Select the model directory
## 4. Load the model from the selected model directory
## 5. Save the model with a new name for neural network
## 6. The user trains the neural network incrementally
## 7. The user tests the neural network with the selected mode
## 8. The user gives feedback to the neural network by selecting the correct or incorrect prediction
## 9. By using models, the learning stage is loaded.
## 10. The unknown image is grouped into clusters to form a new class.
## 11. By using the clustering method, the unknown image is classified and asked the use for label.
## 12. If the label of the image is found as vehicle, the image is processed to find the license plate.
## 13. The license plate is read and the text is shown to the user.
## To sum up, image processing, neural network, clustering, and text recognition are used in the project. 

# Import the required libraries
import os, sys, time, subprocess, platform

# Function to install the required packages
def install_requirements(requirements_file=None):
    if requirements_file is None:
        return 
    if not os.path.isfile(requirements_file):
        raise FileNotFoundError(f"{requirements_file} not found")

    system = platform.system()
    if system == 'Windows':
        subprocess.check_call(['pip', 'install', '-r', requirements_file])
    elif system in ['Linux', 'Darwin']:
        subprocess.check_call(['pip3', 'install', '-r', requirements_file])
    else:
        raise OSError(f"Unsupported operating system: {system}")

# Install the required packages
install_packages = False
# Check if the install_packages flag is set to True
if install_packages:
    # Install the required packages
    install_requirements(r"C:\Dev\github\me536\Project\Main\requirements.txt")

# General use libraries
import cv2, numpy as np, matplotlib.pyplot as plt, tkinter as tk
import pytesseract
# Specific libraries
from tkinter import ttk
from PIL import Image, ImageTk
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# Set the environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import the required libraries for neural network
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Neural Network class for the project
class NeuralNetwork:
    def __init__(self, input_shape):
        # Initialize the input shape
        self.input_shape = input_shape
        # Initialize the all class labels
        self.all_class_labels = []
        # Initialize the models variable
        self.models = []
        # Initialize the class labels
        self.class_labels = None
        # Unknown images for clustering
        self.unknown_images = []

    # Function to create the model
    def create_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),  # Explicitly define the input shape
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.class_labels), activation='softmax')
        ])

        model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
        return model

    # Function to get all class labels for all models
    def get_all_class_labels(self):
        # Initialize the all class labels
        all_class_labels = []
        # Iterate through all class labels
        for class_labels in self.all_class_labels:
            # Append the class labels to the all class labels
            all_class_labels += class_labels
        # Remove duplicates
        all_class_labels = list(set(all_class_labels))
        # Return the all class labels
        return all_class_labels

    # Function to train the model
    def load_model(self, model_path, model_name):
        if model_name == "":
            return
        class_labels_txt = model_name.split(".")[0] + ".txt"
        class_labels_path = os.path.join(model_path, class_labels_txt)
        with open(class_labels_path, "r") as f:
            class_labels = f.read().splitlines()
        self.class_labels = eval(class_labels[0])
        model_path = os.path.join(model_path, model_name)
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()
        # Append the class labels to the all class labels
        self.all_class_labels.append(self.class_labels)
        # Append the model to the models variable
        self.models.append(self.model)

    # Function to preprocess images and labels
    def load_images_and_labels(self, image_paths, labels, img_size=(128, 128)):
        images = []
        for path in image_paths:
            try:
                img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img)
            except Exception as e:
                print(f"An error occurred: {e}")
        return np.array(images), np.array(labels)

    # Function to augment images
    def augment_images(self, X_train, y_train, batch_size=32, img_size=(128, 128)):
        datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
        datagen.fit(X_train)
        return datagen.flow(X_train, y_train, batch_size=batch_size)

    # Incremental training function
    def train_incrementally(self, new_image_paths, new_labels, class_labels, img_size=(128, 128), epochs=10):
        # If class labels is None, define it with 0 to n-1
        if class_labels is None:
            class_labels = np.sort(np.unique(new_labels))
        self.class_labels = class_labels
        # Initialize the model by calling the create_model function
        self.model = self.create_model()
        # Get the model variable
        model = self.model
        # Load the new images and labels
        X_new, y_new = self.load_images_and_labels(new_image_paths, new_labels, img_size)

        # Split the new data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

        # Train the model on the new data
        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

        # Update the model variable
        self.model = model

        # Add the new class labels to the all class labels
        self.all_class_labels.append(class_labels)
        # Add the model to the models variable
        self.models.append(self.model)

    # Function to save the model
    def save_model(self, model_dir, model_name):
        cmd_message = ""
        model_name = model_name + ".keras"
        model_path = os.path.join(model_dir, model_name)
        class_labels_txt = model_name.split(".")[0] + ".txt"
        class_labels_path = os.path.join(model_dir, class_labels_txt)
        try:
            self.model.save(model_path)
            cmd_message = f"Model saved successfully at {model_path}"
            self.model.summary()
        except Exception as e:
            cmd_message = f"An error occurred: {e}"
        try:
            with open(class_labels_path, "w") as f:
                f.write(str(self.class_labels))
        except Exception as e:
            cmd_message = f"An error occurred: {e}"
        return cmd_message

    # Function to preprocess the image for prediction
    def preprocess_image(self, image_path, input_shape):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found at the specified path.")

        image = cv2.resize(image, input_shape[:2])  # Resize to model input size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    # Function to make a prediction
    def predict_image(self, image_path):
        # Define the class labels
        class_labels = None
        # Define the predicted label and confidence
        predicted_labels = {}
        confidences = {}
        feature_vectors = []
        try:
            # Iterate through all the models
            for i, model in enumerate(self.models):
                # Get the class labels
                class_labels = self.all_class_labels[i]
                # Check if the model is loaded
                if model is None or class_labels is None:
                    continue
                # Preprocess the input image
                preprocessed_image = self.preprocess_image(image_path, self.input_shape)
                # Get model feature vector
                feature_vector = (model.predict(preprocessed_image))[0]
                # If the size of the feature vector is 2, and the second class label begins with "not", then take the first class label
                if len(feature_vector) == 2:
                    if class_labels[1].startswith("not"):
                        # Remove the second class label's feature vector
                        feature_vector = np.array([feature_vector[0]])
                        # Remove the second class label
                        class_labels = [class_labels[0]]
                # Get the class with the highest probability
                predicted_class_index = np.argmax(feature_vector)
                predicted_label = class_labels[predicted_class_index]
                confidence = feature_vector[predicted_class_index]
                # Append the predicted label and confidence to the lists
                predicted_labels[predicted_label] = confidence
                confidences[predicted_label] = confidence
                # Append the feature vector to the list
                feature_vector = feature_vector.tolist()
                feature_vectors.append(feature_vector)

            # Get the predicted label with the highest confidence
            predicted_label = max(predicted_labels, key=predicted_labels.get)
            confidence = confidences[predicted_label]

            return predicted_label, confidence, feature_vectors

        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None, None

    # Function to classify the image
    def classify_image(self, image_path):
        # Model comment variable
        model_comment = ""
        # Get the predicted label, confidence, and feature_vectors
        predicted_label, confidence, feature_vectors = self.predict_image(image_path)
        # If the predicted label is None, return
        if predicted_label is None:
            return None, None, None
        # If the confidence is less than 0.5, add the image to the unknown images
        if confidence < 0.5:
            self.unknown_images.append(image_path)
            model_comment = "Unknown"
            predicted_label = "Unknown"
        # Elif the confidence is less than 0.7, marked the prediction as unsure
        elif confidence < 0.7:
            model_comment = "Unsure"
        # Else, the prediction is correct
        else:
            model_comment = "Correct"

        return predicted_label, confidence, feature_vectors, model_comment

# License Plate Detector class for the project
class LicensePlateDetector:
    def __init__(self, image_path, show=False, language="tr", debug=False):
        # First enable the Tesseract executable path
        pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Abdulkadir\AppData\Local\Tesseract-OCR\tesseract.exe" # Update with your Tesseract path
        # Initialize the image variable
        self.image = None
        # Initialize the image path variable
        self.image_path = image_path
        # Initialize the license plate, coord, text variables
        self.plate = None
        self.coords = None
        self.text = None
        # Initialize the debugging and show flag
        self.debug = debug
        self.show = show
        # Initialize the language
        self.language = language

    def detect_license_plate(self):
        # Convert image to gray scale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Use Canny Edge Detection
        edges = cv2.Canny(gray, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Define a list to store possible license plates
        plates = []
        for contour in contours:
            # Approximate the contour to a rectangle
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Look for a quadrilateral
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 2 < aspect_ratio < 6:  # Filter for license plate-like rectangles
                    plate = self.image[y:y+h, x:x+w]
                    plates.append((plate, (x, y, w, h)))
        return plates

    def recognize_plate_text(self, plate_image):
        # Preprocess the license plate image
        gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        _, binary_plate = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)
        
        # Run OCR on the processed plate image
        text = pytesseract.image_to_string(binary_plate, config="--psm 7")
        return text.strip()

    def detector_result(self):
        # Load the image
        self.image = cv2.imread(self.image_path)
        # Detect the license plate
        plates = self.detect_license_plate()
        if plates is None or len(plates) == 0:
            return

        # Recognize text from the plate
        for plate_info in plates:
            plate, coords = plate_info
            text = self.recognize_plate_text(plate)
            if len(text) >= 5:  # Filter out false positives
                if self.language == "tr":
                    try:
                        _ = int(text.strip()[0])
                        break
                    except:
                        plate = None
                        text = ""
                        coords = None
                elif text.strip()[0] in list("0123456789ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"):
                    break
            else:
                plate = None
                text = ""
                coords = None
        self.plate = plate
        self.text = text
        self.coords = coords

        if self.show:
            self.show_image()

        return plate, text, coords
    
    def show_image(self):
        # Display the result
        image = cv2.imread(self.image_path)
        if self.coords is not None:
            x, y, w, h = self.coords
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, self.text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Result", image)
        if self.plate is not None:
            cv2.imshow("License Plate", self.plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Clustering class for the project
class Clustering:
    def __init__(self, data, k=-1):
        # Define data
        self.data = data
        # Run the KMeans clustering algorithm
        best_k, labels, kmeans, metrics = self.find_optimal_k(k)
        # Define the output variables
        self.best_k = best_k
        self.labels = labels
        self.kmeans = kmeans
        self.metrics = metrics

    # Support functions for clustering analysis
    def plot_data_points(self, labels, kmeans, ax=None):
        """
        Plot the data points with colored clusters.

        Parameters:
        - data: np.ndarray, shape (n_samples, n_features), raw data points
        - labels: np.ndarray, shape (n_samples,), cluster labels for the data points
        - kmeans: KMeans instance, fitted KMeans model
        - ax: matplotlib.axes.Axes, optional, axis to plot on

        Returns:
        - ax: matplotlib.axes.Axes, axis with the plot
        """
        # Define data
        data = self.data
        # Check the dimensions of the data
        if data.shape[1] != 2:
            return
        # If no axis is provided, create a new figure and axis
        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot the data points with colored clusters
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
        # Plot the cluster centers
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='x')
        
        # Set axis labels
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        # Display the plot
        plt.show()

    # Silhouette and Elbow Method plots
    def plot_metrics(self, metrics):
        """
        Plot the distortion and silhouette scores for different k values.

        Parameters:
        - metrics: dict, dictionary with distortions and silhouette scores for each k
        """
        # Create a new figure and axis
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the distortion
        ax[0].plot(metrics["k_values"], metrics["distortions"], marker='o')
        ax[0].set_xlabel("Number of Clusters (k)")
        ax[0].set_ylabel("Distortion")
        ax[0].set_title("Elbow Method")

        # Plot the silhouette scores
        ax[1].plot(metrics["k_values"], metrics["silhouette_scores"], marker='o')
        ax[1].set_xlabel("Number of Clusters (k)")
        ax[1].set_ylabel("Silhouette Score")
        ax[1].set_title("Silhouette Score")

        # Display the plot
        plt.show()

    # The function takes data and number of clusters, then returns the kmeans instance, cluster labels, metrics
    def kmeans_metrics(self, k):
        # Define data
        data = self.data
        # Reshape the data to be able to use KMeans
        data = data.T if data.shape[0] < data.shape[1] else data
        # Fit the KMeans model
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        # Find the distortion and silhouette score
        distortion = kmeans.inertia_
        silhouette = silhouette_score(data, labels)
        # Return the kmeans instance, cluster labels, distortion, and silhouette score
        return kmeans, labels, distortion, silhouette

    # The function finds the local maximas of the data
    def find_local_maxima(self, dataX, dataY):
        """
        Find the indices of local maxima in a NumPy array.

        Parameters:
        data (numpy.ndarray): The input data array.

        Returns:
        numpy.ndarray: The indices of the local maxima.
        """
        # Ensure the input is a NumPy array
        dataX = np.asarray(dataX)
        dataY = np.asarray(dataY)

        # Identify local maxima
        # We can check if the value is greater than the previous and next values
        local_maxima = (dataY[1:-1] > dataY[:-2]) & (dataY[1:-1] > dataY[2:])
        
        # Add 1 to the indices to account for the offset caused by slicing
        local_maxima_indices = np.where(local_maxima)[0] + 1
        
        # Sort the indices by corresponding y values in descending order
        local_maxima_indices = local_maxima_indices[np.argsort(dataY[local_maxima_indices])[::-1]]
        # Find the corresponding x values
        local_maxima_x = dataX[local_maxima_indices]
        # Find the corresponding y values
        local_maxima_y = dataY[local_maxima_indices]

        if len(local_maxima_indices) == 0:
            return 0, dataX[0], dataY[0]
        else:
            return local_maxima_indices, local_maxima_x, local_maxima_y

    # Function to determine optimal k using the Elbow Method and Silhouette Score
    # The detailed information for Silhouette Score can be found in the following link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    # The function returns the optimal k, cluster labels, KMeans instance, and metrics
    def find_optimal_k(self, max_k=-1):
        """
        Determine the optimal number of clusters using the Elbow Method and Silhouette Score.

        Parameters:
        - data: np.ndarray, shape (2, n_samples), raw data points
        - max_k: int, maximum number of clusters to test

        Returns:
        - k_optimal: Optimal number of clusters
        - labels: Cluster labels for the data points
        - kmeans: Fitted KMeans instance
        - metrics: Dictionary with distortions and silhouette scores for each k
        """
        # Define data
        data = self.data
        # First, we need to determine the maximum number of clusters to test
        if max_k == -1:
            max_k = 10

        # Define arrays to store metrics
        distortions = np.zeros(max_k - 1)
        silhouette_scores = np.zeros(max_k - 1)
        k_values = np.array(range(2, min(max_k, len(data)) + 1))
        
        # Initialize variables to store the best k and corresponding metrics
        best_k = None
        best_labels = None
        best_kmeans = None

        # For each k value, fit the KMeans model and compute metrics
        for i, k in zip(range(len(k_values)), k_values):
            try:
                # Perform KMeans clustering
                kmeans, labels, distortion, silhouette = self.kmeans_metrics(k)
                # Store the metrics
                distortions[i] = distortion
                silhouette = silhouette_score(data, labels)
                silhouette_scores[i] = silhouette
            
            except ValueError as e:
                continue
        
        # The metrics for silhouette score and distortion are returned for analysis
        # However, the best k might be found at the beginning of the range which is not desired
        # Thus, we need to check if any near best k exists with a lower distortion
        # Find peak silhouette scores
        peak_indices, peak_kValues, peak_silhouette_scores = self.find_local_maxima(k_values, silhouette_scores)
        if type(peak_indices) == int:
            best_k = peak_kValues
        else:
            best_k = peak_kValues[0]

        best_kmeans = KMeans(n_clusters=best_k, random_state=42).fit(data)
        best_labels = best_kmeans.labels_
        
        # Ensure we have a valid k
        if best_k is None:
            raise ValueError("Could not determine an optimal k. Check the data or clustering setup.")
        
        # Return metrics for analysis and plotting
        metrics = {
            "k_values": list(k_values),
            "distortions": distortions,
            "silhouette_scores": silhouette_scores,
        }

        return best_k, best_labels, best_kmeans, metrics

    def clustering_results(self):
        return self.best_k, self.labels, self.kmeans, self.metrics

# Main window for user interaction
class Main:
    # Initialize the window
    def __init__(self, root):
        # Initialize the window
        self.root = root
        # Set the title of the window
        self.root.title("ME536 Project - Abdulkadir Sarıtepe")
        # Set the size of the window full screen
        self.root.attributes('-fullscreen', True)

        # Define debugging flag
        self.debugging_flag = False
        # Define Start/Stop flag
        self.start_flag = False
        # Define the unknown image flag
        self.unknown_image_flag = False

        # Define the image label
        self.image_label = None
        # Define the image paths
        self.image_paths = []
        # Define the image index
        self.image_index = 0

        # Correct prediction counter
        self.correct_prediction_counter = 0
        # Incorrect prediction counter
        self.incorrect_prediction_counter = 0

        # CMD message
        self.cmd_message = ""
        # Main directory
        self.main_dir = "C:/Dev/github/me536/Project/Main"
        # Model directory
        self.model_dir = f"{self.main_dir}/Models"
        # Data directory
        self.image_dir = f"C:/Dev/github/Data/Main" # TODO
        # Mode directory
        self.mode_dir = None

        # Define the neural network
        self.nn = NeuralNetwork((128, 128, 3))
        # Define models variable
        self.models = None

        # Define the prediction, confidence and comment variables
        self.prediction = ""
        self.confidence = 0
        self.comment = ""

        # License Plate variable
        self.license_plate = "Not Found"

        # Unknown images
        self.unknown_images = []

        # Frame for input directories
        self.first_frame = tk.Frame(root)
        self.first_frame.pack(pady=5)

        # Project directory
        tk.Label(self.first_frame, text="Project Directory:").grid(row=0, column=0, padx=5, pady=5)
        self.data_dir = tk.Entry(self.first_frame, width=50)
        self.data_dir.insert(0, self.main_dir)
        self.data_dir.grid(row=0, column=1, padx=5, pady=5)

        # Dropdown for mode selection
        tk.Label(self.first_frame, text="Select Mode:").grid(row=0, column=2, padx=5, pady=5)
        self.mode_var = tk.StringVar()
        self.mode_dropdown = ttk.Combobox(self.first_frame, textvariable=self.mode_var)
        self.mode_dropdown['values'] = ('Training', 'Validation', 'Testing', 'Live')
        self.mode_dropdown.grid(row=0, column=3, padx=5, pady=5)

        # Quit button to the right of the window
        self.quit_button = tk.Button(self.first_frame, text="Quit", command=self.quit)
        self.quit_button.grid(row=0, column=4, padx=5, pady=5)
        self.quit_button.config(bg='darkred', fg='white', font=('helvetica', 10, 'bold'))

        # Second frame for information labels
        self.second_frame = tk.Frame(root)
        self.second_frame.pack(pady=5)

        # The user might want to load a model, therefore, a model directory dropdown is provided
        tk.Label(self.second_frame, text="Model Directory:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar()
        self.model_dir_dropdown = ttk.Combobox(self.second_frame, textvariable=self.model_var, width=50)
        self.model_dir_dropdown.grid(row=0, column=1, padx=5, pady=5)

        # Add a refresh button to the right of the model directory dropdown
        self.refresh_button = tk.Button(self.second_frame, text="Refresh", command=self.refresh_model_dir)
        self.refresh_button.grid(row=0, column=2, padx=5, pady=5)

        # Also, the button to load the model is provided
        self.load_model_button = tk.Button(self.second_frame, text="Load Model", command=self.load_model)
        self.load_model_button.grid(row=0, column=3, padx=5, pady=5)

        # Model classes label
        tk.Label(self.second_frame, text="Model Classes:").grid(row=0, column=4, padx=5, pady=5)

        # Add a dropdown menu for looking at the model classes
        self.model_classes_dropdown = ttk.Combobox(self.second_frame, width=30)
        self.model_classes_dropdown.grid(row=0, column=5, padx=5, pady=5)

        # New model name label
        tk.Label(self.second_frame, text="New Model Name:").grid(row=0, column=6, padx=5, pady=5)

        # Take new model name from the user
        self.new_model_name = tk.Entry(self.second_frame, width=50)
        self.new_model_name.insert(0, "Default")
        self.new_model_name.grid(row=0, column=7, padx=5, pady=5)

        # Also, the button to save the model is provided
        self.save_model_button = tk.Button(self.second_frame, text="Save Model", command=self.save_model)
        self.save_model_button.grid(row=0, column=8, padx=5, pady=5)

        # Third frame for information labels
        self.third_frame = tk.Frame(root)
        self.third_frame.pack(pady=5)

        # Correct Prediction Label
        self.correct_prediction_count_label = tk.Label(self.third_frame, text=f"Correct Predictions: {self.correct_prediction_counter}")
        self.correct_prediction_count_label.grid(row=0, column=0, padx=5, pady=5)

        # Incorrect Prediction Label
        self.incorrect_prediction_count_label = tk.Label(self.third_frame, text=f"Incorrect Predictions: {self.incorrect_prediction_counter}")
        self.incorrect_prediction_count_label.grid(row=0, column=1, padx=5, pady=5)

        # Current State Label
        self.current_state_label = tk.Label(self.third_frame, text=f"CMD: {self.cmd_message}")
        self.current_state_label.grid(row=0, column=2, padx=5, pady=5)

        # buttons frame
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=5)

        self.start_button = tk.Button(self.button_frame, text="Start", command=self.start, bg='SystemButtonFace')
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.next_button = tk.Button(self.button_frame, text="Next", command=self.next, state=tk.DISABLED)
        self.next_button.grid(row=0, column=3, padx=5, pady=5)

        self.debugging_button = tk.Button(self.button_frame, text="Debugging", command=self.debugging, bg="darkgray")
        self.debugging_button.grid(row=0, column=4, padx=5, pady=5)

        # Prediction Display Frame
        self.prediction_display_frame = tk.Frame(root)
        self.prediction_display_frame.pack(pady=5)

        # Prediction Label
        self.prediction_label = tk.Label(self.prediction_display_frame, text=f"Prediction: {self.prediction}")
        self.prediction_label.grid(row=0, column=0, padx=5, pady=5)

        # Confidence Label
        self.confidence_label = tk.Label(self.prediction_display_frame, text=f"Confidence: {self.confidence}")
        self.confidence_label.grid(row=0, column=1, padx=5, pady=5)

        # Comment Label
        self.comment_label = tk.Label(self.prediction_display_frame, text=f"Comment: {self.comment}")
        self.comment_label.grid(row=0, column=2, padx=5, pady=5)

        # Prediction Frame
        self.prediction_frame = tk.Frame(root)
        self.prediction_frame.pack(pady=5)

        # Correct Prediction Label
        self.correct_prediction_label = tk.Label(self.prediction_frame, text="Is the prediction correct?")
        self.correct_prediction_label.grid(row=0, column=0, padx=5, pady=5)

        # Correct Prediction Button
        self.correct_prediction_button = tk.Button(self.prediction_frame, text="Yes", command=self.correct_prediction, bg='PaleGreen1', state=tk.DISABLED)
        self.correct_prediction_button.grid(row=0, column=1, padx=5, pady=5)

        # Incorrect Prediction Button
        self.incorrect_prediction_button = tk.Button(self.prediction_frame, text="No", command=self.incorrect_prediction, bg='salmon1', state=tk.DISABLED)
        self.incorrect_prediction_button.grid(row=0, column=2, padx=5, pady=5)

        # License Plate Detector Frame
        self.license_plate_detector_frame = tk.Frame(root)
        self.license_plate_detector_frame.pack(pady=5)

        # License Plate Detector Label
        self.license_plate_detector_label = tk.Label(self.license_plate_detector_frame, text=f"License Plate Detector: {self.license_plate}")
        self.license_plate_detector_label.grid(row=0, column=0, padx=5, pady=5)

        # Image frame
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=5)

        # Image Label
        self.image_label = tk.Label(self.image_frame, image=None)
        self.image_label.pack()

        # Update the model directory
        self.refresh_model_dir()

    # Function to update the model and data directories
    def update_directories(self):
        # Update the model directory
        self.model_dir = f"{self.data_dir.get()}/Models"
        # Update the data directory
        self.image_dir = f"C:/Dev/github/Data/Main" # TODO

    # Function to get all files in the directory
    def get_all_files(self, data_dir):
        # Get JPEG files in the directory
        jpeg_files1 = [f for f in os.listdir(data_dir) if f.endswith('.JPEG')]
        # Get jpeg files in the directory
        jpeg_files2 = [f for f in os.listdir(data_dir) if f.endswith('.jpeg')]
        # Concatenate the lists
        jpeg_files = jpeg_files1 + jpeg_files2
        # Get jpg files in the directory
        jpg_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        # Get png files in the directory
        png_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        # Concatenate the lists
        all_files = jpeg_files + jpg_files + png_files
        return all_files

    # Function to predict the image
    def predict_image(self, image_path):
        # Classify the image
        predicted_label, confidence, predictions, model_comment = self.nn.classify_image(image_path)
        # Update the prediction, confidence, and comment variables
        self.prediction = predicted_label
        self.confidence = round(confidence, 2)
        self.comment = model_comment
        if self.comment == "Unknown":
            self.prediction = "Unknown"
        # Update the prediction label
        self.prediction_label.config(text=f"Prediction: {self.prediction}")
        # Update the confidence label
        self.confidence_label.config(text="Confidence: " + str(self.confidence))
        # Update the comment label
        self.comment_label.config(text=f"Comment: {self.comment}")
        # Return the comment
        return model_comment

    # Function to perform the unknown image procedure
    def unknown_image_procedure(self):
        # Update CMD message
        self.cmd_message = "New Image Detected, Unknown Object."
        self.current_state_label.config(text=f"CMD: {self.cmd_message}")
        # Ask the user for the label of the unknown image with the known classes
        self.correct_prediction_label.config(text="Unknown object detected, is the object on the classes list?")
        self.correct_prediction_button.config(state=tk.NORMAL)
        self.incorrect_prediction_button.config(state=tk.NORMAL)
        # If the user selects yes, the image is added to the dataset
        # If the user selects no, the image is added to the unknown images
    
    # Function to perform the known image procedure
    def known_image(self):
        # Ask for the label of the image
        self.correct_prediction_label.config(text="Please enter the label of the object:")
        # Deactivate the Next button
        self.next_button.config(state=tk.DISABLED)
        # Get the value of the dropdown
        model_classes = self.model_classes_dropdown.get()
        # If the model classes are not selected, return
        if not model_classes:
            self.next_button.config(state=tk.NORMAL)
            return
        # If the model classes are selected, update the prediction
        self.prediction = model_classes
        # Update the prediction label
        self.prediction_label.config(text=f"Prediction: {self.prediction}")
        # Update the confidence label
        self.confidence_label.config(text="Confidence: 1.0")
        # Update the comment label
        self.comment_label.config(text="Comment: Correct")
        # Update the CMD message
        self.cmd_message = "Known Object Detected."
        self.current_state_label.config(text=f"CMD: {self.cmd_message}")
        # Activate the Next button
        self.next_button.config(state=tk.NORMAL)

    # Function to cluster the unknown images
    def unknown_image_clustering(self):
        # Form the data matrix by using the cnn features
        predictions_of_unknown_images = np.zeros((len(self.unknown_images), len(self.nn.class_labels)))
        for i, image_path in zip(range(len(self.unknown_images)), self.unknown_images):
            # Predict the image
            predicted_label, confidence, predictions = self.nn.predict_image(image_path)
            # Convert the predictions to a numpy array
            predictions = np.array(predictions)
            # Append the predictions to the list
            predictions_of_unknown_images[i, :] = predictions
        # Cluster the unknown images
        best_k, best_labels, best_kmeans, metrics = Clustering(predictions_of_unknown_images).clustering()
        #If any cluster has more than 5 images, ask the user for the label
        for i in range(best_k):
            if np.sum(best_labels == i) > 5:
                # Ask the user for the label of the cluster
                self.correct_prediction_label.config(text="Please enter the label of the cluster:")
                # Deactivate the Next button
                self.next_button.config(state=tk.DISABLED)
                # Display the clustered images
                self.show_clustered_images(self.unknown_images[best_labels == i])
                # Get the value of the dropdown
                model_classes = self.model_classes_dropdown.get()
                # If the model classes are not selected, return
                if not model_classes:
                    self.next_button.config(state=tk.NORMAL)
                    return
                # If the model classes are selected, update the prediction
                self.prediction = model_classes
                # Update the prediction label
                self.prediction_label.config(text=f"Prediction: {self.prediction}")
                # Update the confidence label
                self.confidence_label.config(text="Confidence: 1.0")
                # Update the comment label
                self.comment_label.config(text="Comment: Correct")
                # Update the CMD message
                self.cmd_message = "Known Object Added."
                self.current_state_label.config(text=f"CMD: {self.cmd_message}")
                # Activate the Next button
                self.next_button.config(state=tk.NORMAL)
                break

    # Function to load the model
    def load_model(self):
        # Load the model
        model_name = self.model_dir_dropdown.get()
        # If the model directory is not selected, return
        if not self.model_dir:
            return
        # Load the model
        self.nn.load_model(self.model_dir, model_name)
        # Update the model classes dropdown
        self.model_classes_dropdown['values'] = self.nn.get_all_class_labels()

    # Function to start the session
    def save_model(self):
        # Save the model
        model_name = self.new_model_name.get()
        # If the model directory is not selected, return
        if not self.model_dir:
            return
        if model_name == "Default":
            model_name = f"model_{time.strftime('%Y%m%d_%H%M%S')}"
        # Save the model
        cmd_message = self.nn.save_model(self.model_dir, model_name)
        # Update the model directory
        self.refresh_model_dir()
        # Update the CMD message
        self.cmd_message = cmd_message
        self.current_state_label.config(text=f"CMD: {self.cmd_message}")

    # Function to show the image
    def show_image(self, image_path):
        # Read the image
        image = cv2.imread(image_path)
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the image to PIL format
        image = Image.fromarray(image)
        # Convert the image to ImageTk format
        image = ImageTk.PhotoImage(image)
        # If the image label is None, initialize it
        if self.image_label is None:
            self.image_label.image = image
            self.image_label.pack()
        # Otherwise, update the image
        else:
            self.image_label.configure(image=image)
            self.image_label.image = image

    # Function to show the clustered images
    def show_clustered_images(self, image_paths, num_cols=4):
        # Find the number of rows
        num_rows = len(image_paths) // num_cols
        if len(image_paths) % num_cols != 0:
            num_rows += 1
        # Create a plot for the images
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, 20))
        # Combine all the images in the image paths
        for i, image_path in zip(range(len(image_paths)), image_paths):
            # Read the image
            image = cv2.imread(image_path)
            # Convert the image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Plot the image
            ax[i // num_cols, i % num_cols].imshow(image)
            ax[i // num_cols, i % num_cols].axis('off')
        # Add the title
        fig.suptitle("Clustered Images", fontsize=20)
        # Save the plot as a jpg file
        try:
            clustered_images_dir_path = f"{self.main_dir}/Clusters/Custered_Images.jpg"
            plt.savefig(clustered_images_dir_path)
        except:
            pass
        # Show the image on the clustered images path in the image label
        self.show_image(clustered_images_dir_path)

    # Function to debug the application
    def debugging(self):
        # Toggle the debugging flag
        self.debugging_flag = not self.debugging_flag
        # If the debugging flag is true, set the color of the debugging button to red
        if self.debugging_flag:
            self.debugging_button.config(bg='red')
        # If the debugging flag is false, set the color of the debugging button to white
        else:
            self.debugging_button.config(bg='white')

    # Function to next the image
    def next(self):
        # If the session is not started, return
        if not self.start_flag:
            return
        # Default labels for correct and incorrect predictions
        self.correct_prediction_label.config(text="Is the prediction correct?")
        # Increment the image index
        self.image_index += 1
        # If the image index is greater than the number of images, reset the image index
        if self.image_index >= len(self.image_paths):
            self.image_index = 0

        # Find the license plate
        self.license_plate_class = LicensePlateDetector(f'{self.mode_dir}/{self.image_paths[self.image_index]}', show=False, language="en", debug=self.debugging_flag)
        try:
            self.license_plate = self.license_plate_class.detector_result()[1]
        except:
            self.license_plate = "Not Found"
        if self.license_plate == "" or self.license_plate is None:
            self.license_plate = "Not Found"
        self.license_plate_detector_label.config(text=f"License Plate Detector: {self.license_plate}")
        # Predict the image
        comment = self.predict_image(f'{self.mode_dir}/{self.image_paths[self.image_index]}')
        # If the comment is unknown, rise the unknown image flag
        if comment == "Unknown":
            self.unknown_image_flag = True
            self.unknown_image_procedure()
        # Show the image
        self.show_image(f'{self.mode_dir}/{self.image_paths[self.image_index]}')
        # Deactivate the Next button
        self.next_button.config(state=tk.DISABLED)
        # Activate the Correct and Incorrect buttons
        self.correct_prediction_button.config(state=tk.NORMAL)
        self.incorrect_prediction_button.config(state=tk.NORMAL)

    # Function to show correct prediction
    def correct_prediction(self):
        if self.unknown_image_flag:
            # Add the image to the dataset
            self.unknown_image_flag = False
            self.cmd_message = "The detection is incorrect, the image is added to the dataset."
            self.current_state_label.config(text=f"CMD: {self.cmd_message}")
            # Deactivate the Correct and Incorrect buttons
            self.next_button.config(state=tk.NORMAL)
            self.correct_prediction_button.config(state=tk.DISABLED)
            self.incorrect_prediction_button.config(state=tk.DISABLED)
            self.known_image()
        else:
            self.correct_prediction_counter += 1
            self.cmd_message = "Correct Prediction"
            self.current_state_label.config(text=f"CMD: {self.cmd_message}")
            self.correct_prediction_count_label.config(text=f"Correct Predictions: {self.correct_prediction_counter}")
            # Deactivate the Correct and Incorrect buttons
            self.correct_prediction_button.config(state=tk.DISABLED)
            self.incorrect_prediction_button.config(state=tk.DISABLED)
            # Activate the Next button
            self.next_button.config(state=tk.NORMAL)

    # Function to show incorrect prediction
    def incorrect_prediction(self):
        if self.unknown_image_flag:
            # Add the image to the unknown images
            self.unknown_image_flag = False
            self.cmd_message = "Unknown Image Added to the Unknown Images"
            self.current_state_label.config(text=f"CMD: {self.cmd_message}")
            # Deactivate the Correct and Incorrect buttons
            self.next_button.config(state=tk.NORMAL)
            self.correct_prediction_button.config(state=tk.DISABLED)
            self.incorrect_prediction_button.config(state=tk.DISABLED)
            self.unknown_images.append(f'{self.mode_dir}/{self.image_paths[self.image_index]}')
            # Cluster the unknown images
            if len(self.unknown_images) > 5:
                self.unknown_image_clustering()
        else:
            self.incorrect_prediction_counter += 1
            self.cmd_message = "Incorrect Prediction"
            self.current_state_label.config(text=f"CMD: {self.cmd_message}")
            self.incorrect_prediction_count_label.config(text=f"Incorrect Predictions: {self.incorrect_prediction_counter}")
            # Deactivate the Correct and Incorrect buttons
            self.correct_prediction_button.config(state=tk.DISABLED)
            self.incorrect_prediction_button.config(state=tk.DISABLED)
            # Activate the Next button
            self.next_button.config(state=tk.NORMAL)

    # Function to refresh the model directory
    def refresh_model_dir(self):
        # Update the all directories
        self.update_directories()
        # Refresh the model directory
        self.model_dir_dropdown['values'] = [f"{f}" for f in os.listdir(self.model_dir) if (f.endswith('.pt') or f.endswith('.pth') or f.endswith('.keras'))] + ['All']
        # Refresh the model classes dropdown
        self.model_classes_dropdown['values'] = self.nn.class_labels

    # Function to start the session
    def start(self):
        if self.mode_var.get() == "":
            self.cmd_message = "Please select a mode"
            self.current_state_label.config(text=f"CMD: {self.cmd_message}")
            return
        # Toggle the start flag
        self.start_flag = not self.start_flag
        # If the start flag is true, set the color of the start button to green
        if self.start_flag:
            # Set the color of the start button to green
            self.start_button.config(bg='red')
            # Activate the Previous, Play and Next buttons
            if self.mode_var.get() != 'Live':
                self.next_button.config(state=tk.NORMAL)
            # Perform the action
            # Get the selected mode
            mode = self.mode_var.get()
            # Update the directories
            self.update_directories()
            if mode != 'Live':
                # Get the data directory
                image_dir = self.image_dir
                # Get the mode data directory
                self.mode_dir = f"{image_dir}/{mode}"
                self.image_paths = self.get_all_files(self.mode_dir)

                if not self.image_paths:
                    print("No images found in the directory")
                    self.next_button.config(state=tk.DISABLED)
                    self.start_button.config(bg='SystemButtonFace')
                    self.start_flag = False
                    self.cmd_message = "No images found in the directory"
                    self.current_state_label.config(text=f"CMD: {self.cmd_message}")
                    return

                # Find the license plate
                self.license_plate_class = LicensePlateDetector(f"{self.mode_dir}/{self.image_paths[self.image_index]}", show=False, language="en", debug=self.debugging_flag)
                try:
                    self.license_plate = self.license_plate_class.detector_result()[1]
                except:
                    self.license_plate = "Not Found"
                if self.license_plate == "" or self.license_plate is None:
                    self.license_plate = "Not Found"
                self.license_plate_detector_label.config(text=f"License Plate Detector: {self.license_plate}")
                # Predict the image
                comment = self.predict_image(f'{self.mode_dir}/{self.image_paths[self.image_index]}')
                # If the comment is unknown, rise the unknown image flag
                if comment == "Unknown":
                    self.unknown_image_flag = True
                    self.unknown_image_procedure()
                # Show the image
                self.show_image(f'{self.mode_dir}/{self.image_paths[self.image_index]}')
                # Deactivate the Next button
                self.next_button.config(state=tk.DISABLED)
                # Activate the Correct and Incorrect buttons
                self.correct_prediction_button.config(state=tk.NORMAL)
                self.incorrect_prediction_button.config(state=tk.NORMAL)
                
            self.start_button.config(text="Stop")
        # If the start flag is false, set the color of the start button to default
        else:
            # Set the color of the start button to default
            self.start_button.config(bg='SystemButtonFace')
            # Also, if the mode is Live, release and close the camera
            self.start_button.config(text="Start")
            # Deactivate the Next buttons
            self.next_button.config(state=tk.DISABLED)
            # Clear the image label
            self.image_label.configure(image='')
            # Clear the image paths
            self.image_paths = []
            # Reset the image index
            self.image_index = 0
            # Reset the correct prediction counter
            self.correct_prediction_counter = 0
            # Reset the incorrect prediction counter
            self.incorrect_prediction_counter = 0
            # Reset the correct prediction label
            self.correct_prediction_count_label.config(text=f"Correct Predictions: {self.correct_prediction_counter}")
            # Reset the incorrect prediction label
            self.incorrect_prediction_count_label.config(text=f"Incorrect Predictions: {self.incorrect_prediction_counter}")
            # Reset the CMD message
            self.cmd_message = ""
            # Reset the current state label
            self.current_state_label.config(text=f"CMD: {self.cmd_message}")

    # Function to quit the application
    def quit(self):
        # Quit the application
        self.root.quit()

# Main function
if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    # Initialize the main window
    app = Main(root)
    # Run the main loop
    root.mainloop()

