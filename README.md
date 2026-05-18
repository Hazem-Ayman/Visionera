# Vision GUI - Face Identification System

A Flask-based web application for real-time face detection and identification using `facenet-pytorch` (MTCNN and InceptionResnetV1) and `mediapipe`.

## Features
- **Real-time Webcam Streaming:** Perform face detection or identification in real-time.
- **Photo & Video Uploads:** Upload images and videos to detect and identify faces.
- **Dynamic Face Registration:** Add new persons to the dataset directly from the web interface on the fly.
- **High Performance:** Implements background threading and frame-skipping for smooth processing and maintaining high FPS.

## Technologies Used
- Python 3
- Flask
- OpenCV
- PyTorch & `facenet-pytorch`
- MediaPipe
- HTML/CSS/JavaScript (Frontend)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your_github_repo_url>
   cd "Vision GUI"
   ```

2. **Install dependencies:**
   Ensure you have Python installed. Install the required packages using pip:
   ```bash
   pip install flask opencv-python mediapipe numpy torch torchvision facenet-pytorch
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```
   The application will start on `http://127.0.0.1:5000/`.

## Usage
- **Raw Mode:** View the webcam feed without any processing.
- **Detection Mode:** Detect faces using MediaPipe with bounding boxes.
- **Identification Mode:** Identify faces using the trained Facenet PyTorch model.
- **Add Person:** Register new people by uploading their photos along with their names. This dynamically updates the recognition data (`encodings.pickle`).

## Directory Structure
- `app.py`: Main Flask application handling routes, background threads, and streaming logic.
- `face_identifier.py`: Standalone face identification script.
- `face_recognetion.py`: Previous iteration script using the `face_recognition` library.
- `templates/`: HTML templates for the frontend web interface.
- `dataset/`: Contains uploaded face images categorized by person.
- `uploads/`: Temporary storage for uploaded video files.
