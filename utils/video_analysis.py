import os
import tempfile
import numpy as np

# Check available computer vision libraries
OPENCV_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None

def get_cv_status() -> dict:
    """Return status of installed computer vision libraries."""
    backend = "None (Manual / Demo Mode)"
    if MEDIAPIPE_AVAILABLE:
        backend = "MediaPipe Face Detection"
    elif OPENCV_AVAILABLE:
        backend = "OpenCV Haar Cascade (Local Built-in)"
        
    return {
        "opencv_available": OPENCV_AVAILABLE,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "active_backend": backend
    }

def detect_faces_in_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Detect faces using MediaPipe if available, else OpenCV Haar Cascade.
    Returns: (annotated_rgb_frame, face_count)
    """
    if frame_bgr is None:
        return np.zeros((100, 100, 3), dtype=np.uint8), 0
        
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if OPENCV_AVAILABLE else frame_bgr
    h, w = frame_rgb.shape[:2]
    faces_detected = 0
    annotated = frame_rgb.copy()
    
    # 1. Try MediaPipe if installed
    if MEDIAPIPE_AVAILABLE:
        try:
            mp_face_detection = mp.solutions.face_detection
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(annotated)
                if results.detections:
                    faces_detected = len(results.detections)
                    for det in results.detections:
                        bboxC = det.location_data.relative_bounding_box
                        x = int(bboxC.xmin * w)
                        y = int(bboxC.ymin * h)
                        box_w = int(bboxC.width * w)
                        box_h = int(bboxC.height * h)
                        # Draw bounding box
                        if OPENCV_AVAILABLE:
                            cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), (59, 130, 246), 2) # Blue box
                            cv2.putText(annotated, f"Face ({det.score[0]:.2f})", (x, max(15, y - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (59, 130, 246), 1)
            return annotated, faces_detected
        except Exception:
            pass # Fall through to OpenCV
            
    # 2. Try OpenCV Haar Cascade
    if OPENCV_AVAILABLE:
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )
            faces_detected = len(faces)
            for (x, y, fw, fh) in faces:
                # Blue bounding box with label
                cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (37, 99, 235), 2)
                cv2.putText(annotated, "Detected Face", (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (37, 99, 235), 1)
            return annotated, faces_detected
        except Exception:
            pass

    return annotated, faces_detected

def process_video_file(file_bytes: bytes, max_samples: int = 8) -> dict:
    """
    Lightweight video analysis pipeline.
    Saves bytes to temporary file, reads metadata, samples evenly spaced frames,
    performs face detection, and computes statistics.
    """
    if not OPENCV_AVAILABLE:
        return {
            "success": False,
            "error": "OpenCV is not installed. Please install opencv-python or use Demo Mode.",
            "frames": [],
            "stats": {}
        }
        
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name
            
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return {
                "success": False,
                "error": "Could not decode video file. Format may be unsupported or corrupted.",
                "frames": [],
                "stats": {}
            }
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        duration_sec = round(total_frames / fps, 1) if fps > 0 and total_frames > 0 else 0.0
        
        # Decide frame sampling indices
        if total_frames <= max_samples or total_frames <= 0:
            sample_indices = list(range(max(1, total_frames)))
        else:
            step = total_frames // max_samples
            sample_indices = [i * step for i in range(max_samples)]
            
        sampled_frames = []
        frames_with_faces = 0
        total_faces_count = 0
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            time_offset_sec = round(idx / fps, 1) if fps > 0 else 0.0
            annotated_frame, face_count = detect_faces_in_frame(frame)
            
            if face_count > 0:
                frames_with_faces += 1
            total_faces_count += face_count
            
            sampled_frames.append({
                "frame_index": idx,
                "time_sec": time_offset_sec,
                "image": annotated_frame,
                "face_count": face_count
            })
            
        cap.release()
        
        analyzed_count = len(sampled_frames)
        detection_rate = round((frames_with_faces / analyzed_count * 100), 1) if analyzed_count > 0 else 0.0
        avg_faces = round(total_faces_count / analyzed_count, 1) if analyzed_count > 0 else 0.0
        
        return {
            "success": True,
            "metadata": {
                "width": width,
                "height": height,
                "fps": round(fps, 1),
                "total_frames": total_frames,
                "duration_seconds": duration_sec
            },
            "stats": {
                "frames_analyzed": analyzed_count,
                "frames_with_faces": frames_with_faces,
                "total_faces_detected": total_faces_count,
                "avg_faces_per_frame": avg_faces,
                "optical_presence_rate": detection_rate
            },
            "frames": sampled_frames,
            "backend": get_cv_status()["active_backend"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error during video processing: {str(e)}",
            "frames": [],
            "stats": {}
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

def generate_synthetic_demo_video_analysis() -> dict:
    """Provides instant realistic sample video analysis when no file is uploaded."""
    import cv2
    
    # Generate 6 synthetic sample frames with simulated faces
    demo_frames = []
    face_counts = [5, 6, 6, 7, 5, 6]
    
    for i, count in enumerate(face_counts):
        # Create a clean classroom canvas
        canvas = np.full((360, 640, 3), 245, dtype=np.uint8)
        # Draw background classroom elements
        if OPENCV_AVAILABLE:
            cv2.rectangle(canvas, (20, 20), (620, 80), (220, 230, 242), -1) # Board
            cv2.putText(canvas, f"Classroom Camera Feed - Frame Sample #{i+1} (t={i*10}s)", 
                        (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 64, 175), 2)
            
            # Draw synthetic student avatars
            for j in range(count):
                cx = 70 + j * 90
                cy = 200 + (j % 2) * 40
                # Draw simulated head and shoulders
                cv2.circle(canvas, (cx, cy), 28, (190, 215, 245), -1)
                cv2.circle(canvas, (cx, cy), 28, (37, 99, 235), 2)
                # Bounding box
                cv2.rectangle(canvas, (cx - 32, cy - 32), (cx + 32, cy + 32), (37, 99, 235), 2)
                cv2.putText(canvas, f"Student {j+1}", (cx - 30, cy - 38), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (37, 99, 235), 1)
        
        demo_frames.append({
            "frame_index": i * 250,
            "time_sec": i * 10.0,
            "image": canvas,
            "face_count": count
        })
        
    return {
        "success": True,
        "is_demo": True,
        "metadata": {
            "width": 640,
            "height": 360,
            "fps": 25.0,
            "total_frames": 1500,
            "duration_seconds": 60.0
        },
        "stats": {
            "frames_analyzed": 6,
            "frames_with_faces": 6,
            "total_faces_detected": sum(face_counts),
            "avg_faces_per_frame": round(sum(face_counts)/6, 1),
            "optical_presence_rate": 100.0
        },
        "frames": demo_frames,
        "backend": "Demo Classroom Simulator"
    }
