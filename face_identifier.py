import cv2
import pickle
import os
import argparse
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device for face_identifier: {device}")

# keep_all=True to detect multiple faces, but for encoding we generally expect one per image
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def encode_faces(dataset_path, encodings_path):
    print(f"[INFO] Quantifying faces from {dataset_path}...")
    known_encodings = []
    known_names = []
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"[INFO] Created dataset directory at '{dataset_path}'.")
        print(f"[INFO] Please add folders named after people and put their images inside, then run this again.")
        return

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                name = os.path.basename(root)
                
                print(f"[INFO] Processing image {image_path} for {name}")
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"[WARNING] Could not read image {image_path}")
                    continue
                    
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face
                boxes, _ = mtcnn.detect(rgb_image)
                if boxes is None:
                    print(f"[WARNING] No face found in {image_path}")
                    continue
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Prevent out-of-bounds crop with margin
                    margin_x = int((x2 - x1) * 0.1)
                    margin_y = int((y2 - y1) * 0.1)
                    
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(image.shape[1], x2 + margin_x)
                    y2 = min(image.shape[0], y2 + margin_y)
                    
                    face_crop = rgb_image[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                        
                    face_crop = cv2.resize(face_crop, (160, 160))
                    
                    face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float() / 255.0
                    face_tensor = (face_tensor - 0.5) / 0.5
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        emb = resnet(face_tensor).cpu().numpy()[0]
                        
                    known_encodings.append(emb)
                    known_names.append(name)
                    
    print(f"[INFO] Serializing encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    with open(encodings_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Encodings saved to {encodings_path} successfully!")

def recognize_faces(video_path, encodings_path):
    print(f"[INFO] Loading encodings from {encodings_path}...")
    if not os.path.exists(encodings_path):
        print(f"[ERROR] Encodings file '{encodings_path}' not found. Please run '--mode encode' first.")
        return
        
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
        
    print(f"[INFO] Starting video stream for {video_path if video_path else 'webcam'}...")
    if video_path == "" or video_path is None or video_path == "0":
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(video_path)
        
    if not video_capture.isOpened():
        print("[ERROR] Could not open video file or webcam.")
        return

    known_encodings = np.array(data["encodings"]) if len(data["encodings"]) > 0 else np.array([])
    known_names = data["names"]

    frame_count = 0
    cached_locations = []
    cached_names = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[INFO] End of video stream.")
            break
            
        frame_count += 1
        
        if frame_count % 3 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            boxes, _ = mtcnn.detect(rgb_small_frame)
            
            locations = []
            names = []
            face_tensors = []
            valid_boxes = []
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Expand bounding box slightly for better cropping
                    margin_x = int((x2 - x1) * 0.1)
                    margin_y = int((y2 - y1) * 0.1)
                    
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(small_frame.shape[1], x2 + margin_x)
                    y2 = min(small_frame.shape[0], y2 + margin_y)
                    
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
                        valid_boxes.append((x1, y1, x2, y2))
                        
                if face_tensors:
                    batch_tensor = torch.stack(face_tensors).to(device)
                    with torch.no_grad():
                        embs = resnet(batch_tensor).cpu().numpy()
                        
                    for i, emb in enumerate(embs):
                        name = "Unknown"
                        if len(known_encodings) > 0:
                            distances = np.linalg.norm(known_encodings - emb, axis=1)
                            best_idx = np.argmin(distances)
                            if distances[best_idx] < 0.75:
                                name = known_names[best_idx]
                        
                        names.append(name)
                        locations.append(valid_boxes[i])
            
            cached_locations = locations
            cached_names = names
            
        for box, name in zip(cached_locations, cached_names):
            x1, y1, x2, y2 = box
            # Scale coordinates back up
            top = y1 * 4
            right = x2 * 4
            bottom = y2 * 4
            left = x1 * 4
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
        cv2.imshow('Face Identifier', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video_capture.release()
    cv2.destroyAllWindows()

def generate_frames(encodings_path):
    if not os.path.exists(encodings_path):
        print(f"[ERROR] Encodings file '{encodings_path}' not found.")
        return
        
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
        
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return

    known_encodings = np.array(data["encodings"]) if len(data["encodings"]) > 0 else np.array([])
    known_names = data["names"]

    frame_count = 0
    cached_locations = []
    cached_names = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % 3 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            boxes, _ = mtcnn.detect(rgb_small_frame)
            
            locations = []
            names = []
            face_tensors = []
            valid_boxes = []
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    margin_x = int((x2 - x1) * 0.1)
                    margin_y = int((y2 - y1) * 0.1)
                    
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(small_frame.shape[1], x2 + margin_x)
                    y2 = min(small_frame.shape[0], y2 + margin_y)
                    
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
                        valid_boxes.append((x1, y1, x2, y2))
                        
                if face_tensors:
                    batch_tensor = torch.stack(face_tensors).to(device)
                    with torch.no_grad():
                        embs = resnet(batch_tensor).cpu().numpy()
                    
                    for i, emb in enumerate(embs):
                        name = "Unknown"
                        if len(known_encodings) > 0:
                            distances = np.linalg.norm(known_encodings - emb, axis=1)
                            best_idx = np.argmin(distances)
                            if distances[best_idx] < 0.75:
                                name = known_names[best_idx]
                        
                        names.append(name)
                        locations.append(valid_boxes[i])
                        
            cached_locations = locations
            cached_names = names
                        
        for box, name in zip(cached_locations, cached_names):
            x1, y1, x2, y2 = box
            top = y1 * 4
            right = x2 * 4
            bottom = y2 * 4
            left = x1 * 4
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    video_capture.release()

def main():
    parser = argparse.ArgumentParser(description="Multiple Person Face Identifier System")
    parser.add_argument("--mode", type=str, choices=["encode", "recognize"], default="recognize",
                        help="Mode: 'encode' to extract features from images, 'recognize' to run on video.")
    parser.add_argument("--dataset", type=str, default="dataset",
                        help="Path to the dataset folder containing images organized by name folders.")
    parser.add_argument("--encodings", type=str, default="encodings.pickle",
                        help="Path to save or load the serialized list of facial encodings.")
    parser.add_argument("--video", type=str, default="", 
                        help="Path to the custom video file. Leave empty to use webcam.")
                        
    args = parser.parse_args()
    
    if args.mode == "encode":
        encode_faces(args.dataset, args.encodings)
    elif args.mode == "recognize":
        recognize_faces(args.video, args.encodings)

if __name__ == "__main__":
    main()

