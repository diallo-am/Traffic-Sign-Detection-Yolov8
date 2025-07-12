# Traffic-Sign-Detection-Yolov8

This project implements a complete road sign detection system that combines computer vision and a graphical dashboard in a single Python script. It can detect traffic signs such as speed limits from images, video, or webcam input, and dynamically updates a visual speedometer or graphical interface based on the detected signs.

🧠 Features
🔍 Real-time road sign detection via webcam or video file

🧠 Pretrained deep learning model (YOLOv8)

🏷️ Classification of common traffic signs (e.g., speed limits, stop)

🖼️ Support for image and video input

💻 Optional integration with a dashboard interface (e.g., speedometer)

📦 Easy to run with Python and OpenCV
✅ Designed for embedded and desktop applications


🚀 Getting Started
🔧 Installation
Clone the repository

Install dependencies

pip install -r requirements.txt
Requires Python 3.7+

▶️ Run Detection


On an image:
python main.py --image images/test_image.jpg


On a video:
python main.py --video videos/test_video.mp4


With a webcam:
python main.py --webcam


 Model Information
The detection model was trained on a dataset of annotated road signs. It supports the following classes:

Speed Limit (10-120 km/h)

Stop

Yield

No Entry
...

Model formats supported: YOLOv8, ONNX, or TensorFlow Lite


📸 GUI Description
The graphical interface is:

Embedded directly within the detection script

Automatically launches on execution

Updates live based on the predicted traffic sign (e.g., a red indicator points to the current speed limit)

📄 License
This project is licensed under the MIT License.
Feel free to use, modify, and contribute!

Contributions
Contributions are welcome! If you find bugs or want to add features, feel free to open a pull request or issue.

