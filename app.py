import os
import time

from flask import Flask, render_template, Response, jsonify, url_for
import cv2
from ultralytics import YOLO
from gtts import gTTS

app = Flask(__name__)

# Ensure static audio directory exists
AUDIO_DIR = os.path.join(app.static_folder, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load YOLO model once
model = YOLO('yolov8n.pt')

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam device 0")


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Single-frame prediction (reduce latency if needed)
        results = model.predict(rgb_frame, conf=0.30, verbose=False)

        # Annotate frame with bounding boxes and labels
        annotated_frame = results[0].plot()

        # Convert back to BGR for OpenCV encoding
        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', bgr_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect', methods=['POST'])
def detect():
    # Read current camera frame quickly
    success, frame = cap.read()
    if not success:
        return jsonify({'error': 'failed to read camera frame'}), 500

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, conf=0.30, verbose=False)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return jsonify({'detected_objects': [], 'audio_url': None})

    labels = []
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        name = model.names.get(cls_id, str(cls_id))
        conf = float(box.conf.item())
        labels.append(f"{name}:{conf:.2f}")

    detected_text = 'Detected: ' + ', '.join(labels)

    # Save as mp3 to static/audio and return link
    filename = f"detect_{int(time.time())}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    tts = gTTS(detected_text, lang='en')
    tts.save(filepath)

    audio_url = url_for('static', filename=f'audio/{filename}', _external=True)

    return jsonify({'detected_objects': labels, 'audio_url': audio_url})


@app.route('/shutdown')
def shutdown():
    cap.release()
    cv2.destroyAllWindows()
    return 'Camera released', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
