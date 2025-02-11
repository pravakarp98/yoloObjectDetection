# YOLO Object Detection

This repository provides a simple setup for running YOLO object detection.  It outlines the steps to create a virtual environment, install necessary dependencies, configure the YOLO model, and execute the object detection script.

## Getting Started

These instructions will guide you through setting up and running the YOLO object detection.

### Prerequisites

Before you begin, ensure you have the following installed:

* Python 3.x
* `venv` (usually included with Python 3)
* `pip` (Python package installer)

### Installation

1. **Create a virtual environment:**
    ```
    python3 -m venv yolo_env
    ```

2. **Activate the virtual environment:**
    ```
    source yolo_env/bin/activate  # On Windows: yolo_env\Scripts\activate
    ```

3. **Install requirements:**
    ```
    pip install -r requirements.txt
    ```

4. **Configure YOLO (yolo-coco):**
    a. Create the directory:
    ```
    mkdir yolo-coco
    ```
    b. Copy coco.names:  Download `coco.names` from `https://github.com/Garima13a/YOLO-Object-Detection/blob/master/data/coco.names` and place it in the yolo-coco directory.

    c. Copy yolov3.cfg: Download `yolov3.cfg` from `https://github.com/Garima13a/YOLO-Object-Detection/blob/master/cfg/yolov3.cfg` and place it in the yolo-coco directory.

    d. Download yolov3.weights: Download the yolov3.weights file (you'll need to find a suitable source for this pre-trained model; the original YOLO website or other reputable sources are good options) and place it in the yolo-coco directory.  (Note: The weights file is usually quite large.)

#### **Running the Object Detection**
Once the setup is complete, you can run the object detection script.  Replace `<your_script_name.py>` with the actual name of your Python script.  You'll likely need to provide arguments specifying the image or video file you want to process, as well as the paths to the configuration and weights files.

```
python <your_script_name.py> --image <path_to_image_or_video> --config yolo-coco/yolov3.cfg --weights yolo-coco/yolov3.weights
```

*(Adjust the command and arguments as needed for your specific script.)*