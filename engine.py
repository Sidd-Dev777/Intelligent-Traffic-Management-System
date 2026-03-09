import cv2
import sqlite3
import time
from ultralytics import YOLO
from database import init_db

model = YOLO('yolov8n.pt')

LOG_COOLDOWN = 5
last_logged = {}

def save_to_db(v_type, conf):
    """Pushes detection data into SQLite with a per-type cooldown."""
    now = time.time()
    if v_type in last_logged and (now - last_logged[v_type]) < LOG_COOLDOWN:
        return
    last_logged[v_type] = now

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
        conn.close()

def process_video_feed():
    """Reads the video feed and runs YOLOv8 inference."""
    cap = cv2.VideoCapture('gate_video.mp4')

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id in [2, 3, 5, 7] and conf > 0.40:
                        vehicle_name = model.names[cls_id]
                        save_to_db(vehicle_name, conf)

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{vehicle_name} {conf:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 2)

            cv2.imshow('KRMU AI Vision Engine', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    init_db()
    process_video_feed()
