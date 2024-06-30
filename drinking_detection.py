## IMPORTS ##
import cv2
import numpy as np
import threading
import pygame
import time

from ultralytics import YOLO

# Load the YOLOv8 model for object detection
model = YOLO('yolov8s.pt')

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize pygame for playing sound
pygame.mixer.init()

# Function to play sound using pygame
def play_sound():
    pygame.mixer.music.load(r"sounds\interface-button-154180.mp3")
    pygame.mixer.music.play()

# Frame count threshold to control sound playback frequency
seconds_threshold = 5  # Adjust this value to set the wait time in seconds between sounds

# Variable to track the last sound play time (initialized to avoid initial wait)
last_sound_time = time.time() - seconds_threshold

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get the current time
    current_time = time.time()

    # Calculate the time elapsed since the last sound was played
    elapsed_time = current_time - last_sound_time
    
    # Detect objects using YOLOv8
    results = model.track(frame, persist=True)
    
    # Initialize lists to store coordinates of detected cups and faces
    cup_coords = []
    face_coords = []
    
    ## PROCESS THE RESULTS RETURNED BY THE MODEL ##
    # Loop through the detection results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get the bounding box coordinates
            conf = box.conf[0]  # Confidence of the detection
            cls = box.cls[0]  # Class of the detected object
            
            # If the detected object is a cup and the confidence is above the threshold
            if conf > 0.3 and model.names[int(cls)] == "cup":
                label = model.names[int(cls)]
                # Draw a rectangle around the detected cup
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Put the label and confidence score above the rectangle
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # Add the center coordinates of the cup to the list
                cup_coords.append(((x1 + x2) / 2, (y1 + y2) / 2))
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # Add the center coordinates of the face to the list
        face_coords.append(((x + x + w) / 2, (y + y + h) / 2))
        
    # Check the distance between detected cups and faces
    if cup_coords and face_coords:
        for face_center in face_coords:
            for cup_center in cup_coords:
                # Calculate the Euclidean distance between the face and cup centers
                distance = np.sqrt((face_center[0] - cup_center[0]) ** 2 + (face_center[1] - cup_center[1]) ** 2)
                
                # If the distance is less than 200 pixels and the elapsed time is greater than the threshold
                if distance < 200 and elapsed_time > seconds_threshold:
                    # Start a new thread to play the sound
                    threading.Thread(target=play_sound).start()
                    # Update the last sound play time
                    last_sound_time = current_time
                    # Display the distance on the frame
                    cv2.putText(frame, str(distance), (int(x), int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Show the frame with the detections
    cv2.imshow('YOLOv8 Tracking', frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
