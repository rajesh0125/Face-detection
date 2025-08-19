<img width="1865" height="899" alt="Screenshot 2025-08-13 000925" src="https://github.com/user-attachments/assets/aa53bf9f-8023-45b8-a50b-f0eaddba275d" />
<img width="1892" height="884" alt="Screenshot 2025-08-13 001155" src="https://github.com/user-attachments/assets/91ed977e-340a-427b-8100-90e4ed4060fd" />


# Face-detection
This AI Face Detection System is a real-time face recognition application built with Python, OpenCV, and Flask. It allows users to register faces, train a recognition model, and detect faces in live video streams through a user-friendly web interface.
📌 Overview
This project is a real-time face detection and recognition system built with:

OpenCV (for face detection & recognition)

Flask (for the web interface)

Haar Cascade & LBPH (for face detection & training)

It allows users to:
✅ Register new faces (collect dataset)
✅ Train a custom face recognizer
✅ Detect & recognize faces in real-time via a web interface

✨ Features
✔ Live Face Detection – Uses Haar Cascade to detect faces in real-time.
✔ Face Recognition – Identifies registered users with LBPH (Local Binary Patterns Histogram).
✔ Web Interface – Built with Flask for easy interaction.
✔ Responsive UI – Works on both desktop and mobile.
✔ Background Image Support – Customizable UI with dynamic background.

🚀 How It Works
Data Collection (project.ipynb)

Captures face samples from the webcam (cv2.VideoCapture).

Saves cropped faces in data/ (e.g., user.1.1.jpg).

Model Training (project.ipynb)

Uses LBPH Face Recognizer to train on collected images.

Saves the trained model as classifier.xml.

Real-Time Detection (app.py)

Flask streams live video with face detection & recognition.

Recognized faces are labeled (e.g., "Rajesh" or "UNKNOWN").

Web Interface (index.html)

Start/Stop buttons to control the video feed.

Smooth animations & responsive design.

🛠 Setup & Installation
1. Clone the Repository
git clone https://github.com/your-username/ai-face-detection.git
cd ai-face-detection
2. Install Dependencies
pip install flask opencv-python numpy pillow
3. Run the Application
python app.py
Open http://localhost:5000 in your browser.
