import cv2
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from PIL import Image
from pytesseract import Output
import matplotlib.pyplot as plt

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
            print("License plate not found!")
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
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
