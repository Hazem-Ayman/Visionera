from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import pickle
import os
import mediapipe as mp
import numpy as np
import io
import threading
import time
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────
current_mode = "raw"          # "raw" | "detection" | "identification"
video_file_path = None        # Path to an uploaded video being streamed
video_file_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
# MediaPipe face detection (used for webcam detection mode)
# ─────────────────────────────────────────────────────────────
mp_faceDetection = mp.solutions.face_detection
face_detection = mp_faceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ─────────────────────────────────────────────────────────────
# Facenet PyTorch initialization
# ─────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device for app: {device}")
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ─────────────────────────────────────────────────────────────
# Pre-load face encodings from training data
# ─────────────────────────────────────────────────────────────
encodings_path = "encodings.pickle"
trained_data = {"encodings": [], "names": []}
if os.path.exists(encodings_path):
    with open(encodings_path, "rb") as f:
        trained_data = pickle.load(f)
else:
    print(f"[WARNING] {encodings_path} not found. Please train first.")

# ─────────────────────────────────────────────────────────────
# Helper: draw bounding boxes + labels on a frame
#   colour = green for known, red for unknown
# ─────────────────────────────────────────────────────────────
GREEN = (0, 255, 0)
RED   = (0, 0, 255)

def _color_for(name: str):
    return GREEN if name != "Unknown" else RED


def apply_detection(frame):
    """Run MediaPipe face detection and draw coloured boxes."""
    H, W, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    if out.detections:
        for det in out.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w  = int(bbox.width * W)
            h  = int(bbox.height * H)
            # Detection mode: we don't know who it is — use green box only
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), GREEN, 3)
    return frame


# ─────────────────────────────────────────────────────────────
# Background-thread face recogniser
#   Recognition is expensive; we offload it to a daemon thread so
#   the streaming loop never blocks waiting for results.
# ─────────────────────────────────────────────────────────────
_id_cache      = {"locations": [], "names": []}  # last result (shared)
_id_cache_lock = threading.Lock()                 # guards _id_cache
_id_pending    = threading.Event()                # signals work is queued
_id_frame_buf  = [None]                           # latest frame for the worker
_id_frame_lock = threading.Lock()                 # guards _id_frame_buf
_id_miss_count = 0                                # consecutive frames with no face
ID_MISS_THRESHOLD = 4   # clear cache only after N consecutive empty detections
IDENTIFY_EVERY_N = 5   # kept for reference


def _draw_identification_results(frame, locations, names, scale=4):
    """Draw bounding boxes + name labels onto frame in-place."""
    font = cv2.FONT_HERSHEY_DUPLEX

    for (top, right, bottom, left), name in zip(locations, names):
        top    *= scale; right *= scale; bottom *= scale; left *= scale
        color   = _color_for(name)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    return frame


def _recognition_worker():
    """Daemon thread: waits for a new frame, runs recognition, stores result."""
    global _id_cache
    while True:
        _id_pending.wait()          # block until a frame is queued
        _id_pending.clear()

        with _id_frame_lock:
            frame = _id_frame_buf[0]
        if frame is None:
            continue

        # Downscale 1/4 for speed
        small     = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes, _ = mtcnn.detect(rgb_small)
        
        locations = []
        names = []
        face_tensors = []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                margin_x = int((x2 - x1) * 0.1)
                margin_y = int((y2 - y1) * 0.1)
                
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(small.shape[1], x2 + margin_x)
                y2 = min(small.shape[0], y2 + margin_y)
                
                # Append in (top, right, bottom, left) format to match existing drawing logic
                locations.append((y1, x2, y2, x1))
                
                # Map coordinates back to original frame size
                orig_x1 = x1 * 4
                orig_y1 = y1 * 4
                orig_x2 = x2 * 4
                orig_y2 = y2 * 4
                
                face_crop = rgb_frame[orig_y1:orig_y2, orig_x1:orig_x2]
                
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, (160, 160))
                    face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
                    face_tensor = (face_tensor - 0.5) / 0.5
                    face_tensors.append(face_tensor)
                else:
                    locations.pop()
                    
            if face_tensors:
                batch_tensor = torch.stack(face_tensors).to(device)
                with torch.no_grad():
                    embs = resnet(batch_tensor).cpu().numpy()
                
                for emb in embs:
                    name = "Unknown"
                    if len(trained_data["encodings"]) > 0:
                        distances = np.linalg.norm(np.array(trained_data["encodings"]) - emb, axis=1)
                        best_idx = np.argmin(distances)
                        if distances[best_idx] < 0.75:
                            name = trained_data["names"][best_idx]
                    names.append(name)

        with _id_cache_lock:
            global _id_miss_count
            if locations:
                # Good detection — update cache and reset miss counter
                _id_miss_count = 0
                _id_cache = {"locations": locations, "names": names}
            else:
                # No faces found this frame — only wipe cache after N consecutive misses
                # so a single bad frame doesn't flash the boxes away
                _id_miss_count += 1
                if _id_miss_count >= ID_MISS_THRESHOLD:
                    _id_cache = {"locations": [], "names": []}


# Start the background recognition thread once at import time
_recognition_thread = threading.Thread(target=_recognition_worker, daemon=True)
_recognition_thread.start()


def apply_identification(frame, submit_to_worker: bool = True):
    """Draw cached recognition results and optionally submit frame to worker.

    submit_to_worker=True  : queues this frame for recognition (does a frame copy).
    submit_to_worker=False : only draws the last cached result — zero extra cost.

    Callers should only set submit_to_worker=True every N frames to avoid
    overwhelming the recognition thread and causing lag.
    """
    if submit_to_worker:
        with _id_frame_lock:
            _id_frame_buf[0] = frame.copy()   # copy only when submitting
        _id_pending.set()                     # wake the worker

    # Drawing cached boxes is always instant
    with _id_cache_lock:
        locs  = list(_id_cache["locations"])
        names = list(_id_cache["names"])

    return _draw_identification_results(frame, locs, names, scale=4)


def identify_frame_sync(frame):
    """Synchronous (blocking) face identification — for photos and one-shot use.

    Runs recognition inline on the calling thread. Never use this in a
    streaming loop; use apply_identification() there instead.
    """
    # Downscale 1/4 for speed
    small     = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(rgb_small)
    
    locations = []
    names = []
    face_tensors = []
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(small.shape[1], x2 + margin_x)
            y2 = min(small.shape[0], y2 + margin_y)
            
            # Append in (top, right, bottom, left) format to match existing drawing logic
            locations.append((y1, x2, y2, x1))
            
            # Map coordinates back to original frame size
            orig_x1 = x1 * 4
            orig_y1 = y1 * 4
            orig_x2 = x2 * 4
            orig_y2 = y2 * 4
            
            face_crop = rgb_frame[orig_y1:orig_y2, orig_x1:orig_x2]
            
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, (160, 160))
                face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
                face_tensor = (face_tensor - 0.5) / 0.5
                face_tensors.append(face_tensor)
            else:
                locations.pop()
                
        if face_tensors:
            batch_tensor = torch.stack(face_tensors).to(device)
            with torch.no_grad():
                embs = resnet(batch_tensor).cpu().numpy()
            
            for emb in embs:
                name = "Unknown"
                if len(trained_data["encodings"]) > 0:
                    distances = np.linalg.norm(np.array(trained_data["encodings"]) - emb, axis=1)
                    best_idx = np.argmin(distances)
                    if distances[best_idx] < 0.75:
                        name = trained_data["names"][best_idx]
                names.append(name)

    return _draw_identification_results(frame, locations, names, scale=4)


# ─────────────────────────────────────────────────────────────
# Webcam streaming generator
# ─────────────────────────────────────────────────────────────
WEBCAM_RECOGNIZE_EVERY = 3   # submit to recognition worker every N webcam frames

def generate_frames():
    global current_mode
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    webcam_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        webcam_frame_idx += 1
        mode = current_mode
        if mode == "detection":
            frame = apply_detection(frame)
        elif mode == "identification":
            # Submit to worker only every N frames; draw cached result on all frames
            submit = (webcam_frame_idx % WEBCAM_RECOGNIZE_EVERY == 0)
            frame = apply_identification(frame, submit_to_worker=submit)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


# ─────────────────────────────────────────────────────────────
# Uploaded-video streaming generator
# ─────────────────────────────────────────────────────────────
VIDEO_RECOGNIZE_EVERY = 4  # submit to recognition worker only every N video frames

def generate_video_file_frames(path: str, mode: str):
    """Stream a video file frame-by-frame with optional face processing.

    Anti-lag strategy:
    - Recognition worker is queued only every VIDEO_RECOGNIZE_EVERY frames.
      All other frames draw the cached result instantly (near-zero cost).
    - Adaptive sleep maintains the video's natural FPS without buffering lag.
    - Frames downscaled to 480 px wide before any CV work.
    - JPEG quality 70 to reduce network payload.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {path}")
        return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_time = 1.0 / fps   # target time per frame

    # Reset recognition cache for a fresh video
    with _id_cache_lock:
        _id_cache["locations"] = []
        _id_cache["names"]     = []

    MAX_WIDTH = 480   # smaller = faster HOG detection inside the worker
    frame_idx = 0

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Downscale to MAX_WIDTH before any processing
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            scale = MAX_WIDTH / w
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_LINEAR)

        if mode == "detection":
            frame = apply_detection(frame)
        elif mode == "identification":
            # Submit a new job every N frames; draw cached result on every frame
            submit = (frame_idx % VIDEO_RECOGNIZE_EVERY == 0)
            frame  = apply_identification(frame, submit_to_worker=submit)

        ret, buffer = cv2.imencode('.jpg', frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')

        # Adaptive sleep: honour the source FPS without busy-waiting
        elapsed = time.time() - t0
        sleep_for = frame_time - elapsed
        if sleep_for > 0.002:    # don't sleep for trivially small amounts
            time.sleep(sleep_for)

    cap.release()


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode
    mode = request.json.get("mode")
    if mode in ["raw", "detection", "identification"]:
        current_mode = mode
    return jsonify({"status": "success", "mode": current_mode})


# ── Photo upload ──────────────────────────────────────────────
@app.route('/process_photo', methods=['POST'])
def process_photo():
    """Receive an image file + desired mode, return the processed JPEG."""
    if 'photo' not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    mode = request.form.get("mode", "detection")
    file_bytes = np.frombuffer(request.files['photo'].read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    if mode == "detection":
        frame = apply_detection(frame)
    elif mode == "identification":
        # Use synchronous processing for photos — async worker would return
        # before the result is ready, giving back a blank (unprocessed) image.
        frame = identify_frame_sync(frame)

    ret, buffer = cv2.imencode('.jpg', frame)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')


# ── Video upload ──────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Save the uploaded video and return the stream URL."""
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    mode = request.form.get("mode", "detection")
    vid  = request.files['video']
    save_path = os.path.join(UPLOAD_FOLDER, "uploaded_video.mp4")
    vid.save(save_path)
    return jsonify({"stream_url": f"/stream_video?mode={mode}", "status": "ok"})


@app.route('/stream_video')
def stream_video():
    """Stream the previously-uploaded video with the requested processing mode."""
    mode = request.args.get("mode", "detection")
    path = os.path.join(UPLOAD_FOLDER, "uploaded_video.mp4")
    if not os.path.exists(path):
        return "No video uploaded yet", 404
    return Response(
        generate_video_file_frames(path, mode),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ── Add Person (Train on-the-fly) ─────────────────────────────
@app.route('/add_person', methods=['POST'])
def add_person():
    """Upload photos of a new person and update encodings.pickle dynamically."""
    name = request.form.get('name', '').strip()
    files = request.files.getlist('photos')
    
    if not name or not files:
        return jsonify({"error": "Name and photos are required"}), 400
        
    person_dir = os.path.join(os.path.dirname(__file__), "dataset", name)
    os.makedirs(person_dir, exist_ok=True)
    
    added_count = 0
    for file in files:
        if file.filename == '':
            continue
            
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            continue
            
        # Save photo to dataset directory
        save_path = os.path.join(person_dir, file.filename)
        cv2.imwrite(save_path, img)
        
        # Encode face
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_img)
        
        if boxes is not None and len(boxes) > 0:
            # We assume one main face per image for training
            x1, y1, x2, y2 = map(int, boxes[0])
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            
            face_crop = rgb_img[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, (160, 160))
                face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
                face_tensor = (face_tensor - 0.5) / 0.5
                face_tensor = face_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    emb = resnet(face_tensor).cpu().numpy()[0]
                    
                trained_data["encodings"].append(emb)
                trained_data["names"].append(name)
                added_count += 1
                
    if added_count > 0:
        # Save the updated encodings back to disk
        with open(encodings_path, "wb") as f:
            pickle.dump(trained_data, f)
        return jsonify({"status": "success", "message": f"Successfully added {added_count} encodings for {name}!"})
    else:
        return jsonify({"error": "No faces could be found in the uploaded photos."}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
