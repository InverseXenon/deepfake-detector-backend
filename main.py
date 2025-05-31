from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import os
import uuid
import gdown

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://deepfake-detector-frontend.vercel.app"  # Update with your Vercel frontend URL after deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "deepfake_detection_model_final.keras"
MODEL_URL = "https://drive.google.com/uc?id=1E_SGbRROP8oAnYC-Mmndi8VtUVm43YEy"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
detector = None

def load_resources():
    global model, detector
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Downloading model from {MODEL_URL}")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully: Input shape: {model.input_shape}, Output shape: {model.output_shape}")
        detector = MTCNN()
        print("MTCNN initialized")
    except Exception as e:
        print(f"Error loading model or MTCNN: {str(e)}")
        model = None
        detector = None

load_resources()

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    try:
        frames = []
        face_confidences = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")
        step = max(1, total_frames // 24)
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(frame)
            print(f"Faces detected in frame {i}: {len(faces)}")
            if faces:
                face = faces[0]
                x, y, w, h = face['box']
                confidence = face['confidence']
                if confidence > 0.9:
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (128, 128))
                    face_img = face_img / 255.0
                    frames.append(face_img)
                    face_confidences.append(confidence)
            if len(frames) >= 24:
                break
        if len(frames) < 10:
            raise ValueError(f"Too few high-confidence faces: {len(frames)}")
        print(f"Selected {len(frames)} frames with average face confidence: {np.mean(face_confidences):.3f}")
        sequences = []
        for i in range(0, min(len(frames), 24), 2):
            sequence = frames[i:i+10]
            if len(sequence) == 10:
                sequences.append(sequence)
            if len(sequences) >= 8:
                break
        if len(sequences) < 8 and frames:
            while len(sequences) < 8:
                sequence = frames[-10:] if len(frames) >= 10 else frames * (10 // len(frames) + 1)[:10]
                sequences.append(sequence)
        sequences = sequences[:8]
        frames_array = np.array(sequences)
        print(f"Sequences shape: {frames_array.shape}")
        return frames_array
    finally:
        cap.release()

@app.get("/")
async def root():
    return {"message": "Deepfake Detector API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in ['mp4']:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an MP4 video")
    file_name = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    print(f"Saving to: {file_path}")
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"File saved at: {file_path}")
        print(f"Processing video: {file_path}")
        frames = preprocess_video(file_path)
        print(f"Input shape to model: {frames.shape}")
        predictions = model.predict(frames)
        print(f"Raw predictions: {predictions}")
        real_probs = predictions[:, 0]
        label = "REAL" if np.mean(real_probs) > 0.5 else "FAKE"
        confidence = np.max(real_probs) if label == "REAL" else np.max(1 - real_probs)
        print(f"Predicted label: {label}, Confidence: {confidence}")
        return {"label": label, "confidence": float(confidence)}
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")