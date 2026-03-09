import cv2
import sqlite3
import time
from flask import Flask, render_template, Response
from ultralytics import YOLO
from database import init_db

app = Flask(__name__)
model = YOLO('yolov8n.pt')

# Cooldown tracker: prevents logging the same vehicle type more than once per N seconds
LOG_COOLDOWN = 5  # seconds between DB writes per vehicle type
last_logged = {}  # {vehicle_type: last_timestamp}

def save_to_db(v_type, conf):
    """Pushes detection data into SQLite with a per-type cooldown to avoid flooding."""
    now = time.time()
    if v_type in last_logged and (now - last_logged[v_type]) < LOG_COOLDOWN:
        return  # skip duplicate within cooldown window
    last_logged[v_type] = now

    conn = None
    try:
        conn = sqlite3.connect('traffic_data.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO vehicle_logs (vehicle_type, confidence_score) VALUES (?, ?)",
            (v_type, conf),
        )
        conn.commit()
    except Exception as e:
        print(f"Database Error: {e}")
    finally:
        if conn:
            conn.close()

FRAME_SKIP = 3  # only run YOLO on every Nth frame for performance

def generate_frames():
    # Make sure you have a video file named gate_video.mp4 in your folder!
    cap = cv2.VideoCapture('gate_video.mp4') 
    
    try: # <-- ADDED THIS to fix the Syntax Error
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop the video
                continue
                
            # Run inference
            results = model(frame, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Lowered confidence to 0.15 for distant vehicles
                    if cls_id in [2, 3, 5, 7] and conf > 0.15:
                        vehicle_name = model.names[cls_id]
                        save_to_db(vehicle_name, conf) # Save to memory
                        
                        # Made the bounding boxes thicker (thickness=3) for better visibility
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # Red boxes for high contrast
                        cv2.putText(frame, f"{vehicle_name} {conf:.2f}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Encode frame for web streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    init_db()  # ensure the table exists before starting
    # Changed to port 5001 and turned off debug to bypass background conflicts
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)