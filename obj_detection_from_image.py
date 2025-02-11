# Import the necessary libraries
import numpy as np
import argparse
import time
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force using X11 instead of Wayland
import cv2
import traceback

# Define paths to YOLO files
weightsPath = 'yolo-coco/yolov3.weights'
configPath = 'yolo-coco/yolov3.cfg'
labelsPath = 'yolo-coco/coco.names'

# Check if YOLO files exist
if not os.path.exists(configPath):
    print(f"Error: Config file not found: {configPath}")
    exit(1)
if not os.path.exists(weightsPath):
    print(f"Error: Weights file not found: {weightsPath}")
    exit(1)
if not os.path.exists(labelsPath):
    print(f"Error: Labels file not found: {labelsPath}")
    exit(1)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Load the COCO class labels
LABELS = open(labelsPath).read().strip().split("\n")
print("LABELS:", LABELS)

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load YOLO model
print("Loading YOLO from disk...")
try:
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    traceback.print_exc()
    exit(1)

# Load the input image and grab its spatial dimensions
image = cv2.imread(args["image"])
if image is None:
    print(f"Error: Unable to load image at {args['image']}")
    exit(1)

(H, W) = image.shape[:2]

# Determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Construct a blob from the input image and perform a forward pass of YOLO
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# Print the time taken for YOLO to process the image
print(f"YOLO took {end - start:.6f} seconds")

# Initialize lists to store bounding boxes, confidences, and class IDs
boxes = []
confidences = []
classIDs = []

# Loop over each layer output
for output in layerOutputs:
    for detection in output:
        # Extract the class ID and confidence
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # Filter out weak detections
        if confidence > args["confidence"]:
            # Scale the bounding box coordinates relative to the image size
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # Calculate the top-left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # Update the lists
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

print(f"Class IDs: {classIDs}")

# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

# Ensure at least one detection exists
if len(idxs) > 0:
    for i in idxs.flatten():
        # Extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # Draw the bounding box and label on the image
        # Check if class ID is within bounds
        if classIDs[i] < len(COLORS):
            color = [int(c) for c in COLORS[classIDs[i]]]
        else:
            print(f"Warning: Class ID {classIDs[i]} is out of bounds.")
            color = [255, 255, 255]  # Default to white if out of bounds
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{LABELS[classIDs[i]]}: {confidences[i]:.4f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()