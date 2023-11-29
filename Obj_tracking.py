from collections import defaultdict
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "./1506.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

pTime = 0
person_cnt = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    cap_h, cap_w, _ = frame.shape # get resolution
    print(cap_w, cap_h, _)

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            #cls_names = results[0].names.values() # get class name
            #print('lsc print', results[0].boxes)
        except AttributeError as err:
            print(err)
            pass

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        #print('lsc print :', results[0].boxes)
        cnt_track_ids = set()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            cnt_track_ids.add(track_id)
            person_cnt = len(cnt_track_ids)
            x, y, w, h = box
            print(float(x), float(y), float(w), float(h))

            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            print(f'{track_id}의 centerpoint 좌표로 발사되었음!')

        # Frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(annotated_frame, f'FPS/COP: {int(fps)}/{person_cnt}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        cv2.line(annotated_frame, (int(cap_w/2), 0), (int(cap_w/2), 1920), (255,0,0), 3)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed~
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
