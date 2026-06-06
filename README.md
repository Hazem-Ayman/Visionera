# 🎥 Visionera - Intelligent Face Detection & Identification System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Machine%20Learning-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange?style=for-the-badge&logo=opencv)](https://opencv.org/)

A powerful Flask-based web application for real-time face detection and identification using cutting-edge deep learning models. Powered by `facenet-pytorch` (MTCNN and InceptionResnetV1) and `mediapipe`.

</div>

---

## ✨ Features

- 🎬 **Real-time Webcam Streaming** - Perform face detection or identification in real-time from your webcam
- 📸 **Photo & Video Processing** - Upload images and videos to detect and identify faces
- 👤 **Dynamic Face Registration** - Add new persons to the dataset directly from the web interface on the fly
- ⚡ **High Performance** - Background threading and frame-skipping for smooth processing and high FPS
- 🎯 **Multiple Detection Modes** - Raw, Detection, and Identification modes for flexibility
- 🔒 **Privacy-Focused** - Local processing with no data sent to external servers

---

## 📸 Project Showcase

![Visionera Main Screen](images/main.png)

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3** | Backend language |
| **Flask** | Web framework |
| **OpenCV** | Video/image processing |
| **PyTorch** | Deep learning framework |
| **facenet-pytorch** | Face detection & embedding |
| **MediaPipe** | Face detection & landmarks |
| **HTML/CSS/JavaScript** | Frontend interface |

---

## 📥 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Hazem-Ayman/Visionera.git
cd Visionera
```

### Step 2: Install Dependencies
```bash
pip install flask opencv-python mediapipe numpy torch torchvision facenet-pytorch
```

### Step 3: Run the Application
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000/`

---

## 🎮 Usage Guide

### **Raw Mode**
View the live webcam feed without any processing.

### **Detection Mode**
Detect faces in real-time using MediaPipe with bounding boxes around detected faces.

### **Identification Mode**
Identify and recognize faces using the trained FaceNet PyTorch model against your registered database.

### **Add Person**
Register new people by uploading their photos along with their names. This dynamically updates the recognition database (`encodings.pickle`).

---

## 📁 Project Structure

```
Visionera/
├── app.py                          # Main Flask application
├── face_identifier.py              # Standalone face identification script
├── face_recognition.py             # Legacy face recognition script
├── images/                         # Project images folder
│   └── main.png                   # Project showcase image
├── templates/                      # HTML templates
├── dataset/                        # Face images categorized by person
├── uploads/                        # Temporary video storage
└── README.md                       # This file
```

---

## 🚀 Quick Start

1. **Start the Application**
   ```bash
   python app.py
   ```

2. **Open Your Browser**
   Navigate to `http://127.0.0.1:5000/`

3. **Choose Your Mode**
   - Select Raw, Detection, or Identification mode
   - Allow webcam access when prompted

4. **Register New Faces**
   - Use the "Add Person" feature to register new individuals
   - Upload their photos and provide their name

5. **Start Identifying**
   - Switch to Identification mode and watch the system recognize faces in real-time!

---

## 🔧 Technical Details

### Face Detection Pipeline
- **MTCNN** (Multi-task Cascaded Convolutional Networks) for initial face detection
- **MediaPipe** for robust face landmark detection
- **FaceNet** for generating face embeddings

### Performance Optimization
- Background threading for non-blocking operations
- Frame skipping to maintain smooth video streaming
- Efficient GPU utilization when available

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Hazem Ayman** - [GitHub Profile](https://github.com/Hazem-Ayman)

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues and submit pull requests.

---

<div align="center">

**Made with ❤️ by Hazem Ayman**

⭐ Don't forget to star the repository if you found it helpful!

</div>
