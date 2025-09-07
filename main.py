import cv2
import math
from ultralytics import YOLO

#Load the Model
model =  YOLO("runs/detect/train/weights/best.pt")


#Load the video capture
videoCap = cv2.VideoCapture('source.mp4')


while True:
    ret, frame =  videoCap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640,384))


    results = model.track(frame,stream=True, tracker="bytetrack.yaml",imgsz=320)

    fighter_positions = {}

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.cls[0])


                track_id = int(box.id[0]) if box.id is not None else -1
                label = f"ID {track_id} | {result.names[cls]} {conf:.2f}"

                #fighter centroid
                cx,cy = int((x1 + x2) /2), int((y1 + y2) /2)
                fighter_positions[track_id] = (cx,cy)

                #Draw bounding box 
                color = (0,255,0) if track_id % 2 == 0 else (255,0,0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label,(x1,y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (cx,cy), 5, color, -1)

            #Calcullate distance if exactly 2 fighter detected
            if len(fighter_positions) == 2:
                ids = list(fighter_positions.keys())
                (x1, y1),(x2,y2) = fighter_positions[ids[0]], fighter_positions[ids[1]]

                distance = math.sqrt((x1-x2) ** 2 + (y1 - y2) ** 2)

                #Draw line between fighters
                cv2.line(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                cv2.putText(frame, f"Dsiatnce:{int(distance)} px",
                            (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                
    cv2.imshow('Combat Sports Analyzer', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()
