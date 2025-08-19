from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Load face classifier and recognizer
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Initialize video capture
video_capture = None

def generate_frames():
    """Generate frames for video feed with face detection and recognition."""
    global video_capture
    while video_capture.isOpened():
        success, img = video_capture.read()
        if not success:
            break
        
        # Perform face detection and recognition
        img = draw_boundary(img, faceCascade, 1.3, 6, (255, 255, 255), "Face", clf)
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        # Yield each frame as a part of a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Serve the video feed."""
    global video_capture
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)  # Start the webcam
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/quit')
def quit():
    """Stop the video feed."""
    global video_capture
    if video_capture is not None:
        video_capture.release()  # Release the webcam
        video_capture = None
    return '', 204  # No content response

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    """Draw rectangles around detected faces and recognize them."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    
    # Iterate through detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # Draw a rectangle
        
        # Recognize the face
        id, pred = clf.predict(gray_img[y:y + h, x:x + w])
        confidence = int(100 * (1 - pred / 300))  # Calculate confidence

        if confidence > 70:
            # Map recognized face to name
            name = "UNKNOWN"
            if id == 1:
                name = "Rajesh"
            elif id == 2:
                name = "Siddharth"
            elif id == 3:
                name = "Anik"
            elif id == 4:
                name = "Dhiraj"
            elif id == 5:
                name = "Deepak"
            elif id == 6:
                name = "Neeraj"
            
            # Annotate the image with the name
            cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            # Mark as unknown if confidence is low
            cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    return img

if __name__ == '__main__':
    app.run(debug=True)
