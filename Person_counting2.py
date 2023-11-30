import cv2
import time
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO

font = cv2.FONT_HERSHEY_DUPLEX
model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture("./video.mp4")

prev_frame_time = 0
new_frame_time = 0

region1=np.array([(0,130),(0,110),(640,260),(640,280)])
region1 = region1.reshape((-1,1,2))

region2=np.array([(0,130),(0,150),(640,300),(640,280)]) # 좌상, 좌하, 우하, 우상
region2 = region2.reshape((-1,1,2))

total_ust = set() # 상위지역 id
total_alt = set() # 하위지역 id
first_in = set() # 들어오는 사람 id
first_out = set() # 나가는 사람 id

while True:
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # print(fps)
    ret, frame = camera.read()
    cap_h, cap_w, _ = frame.shape

    if not ret:
        break
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 선 그리기
    cv2.line(frame, (0,130), (640,280), (255,0,255), 3)
    # 트랙킹 모드
    results = model.track(rgb_img, persist=True, verbose=False)
    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i] # x1, y1 왼쪽 상단 / x2, y2 오른쪽 하단
        score = results[0].boxes.conf[i]
        cls = results[0].boxes.cls[i] # 객체가 속한 클래스
        ids = results[0].boxes.id[i] # 인스턴스의 id
        # 값 변환
        x1, y1, x2, y2, score, cls, ids = int(x1), int(y1), int(x2), int(y2), float(score), int(cls), int(ids)

        if score < 0.5:
            continue
        if cls != 0: # person만
            continue

        # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2) # 바운딩 박스 그리기
        # 객체의 중앙 찾기
        cx = int(x1 / 2 + x2 / 2)
        cy = int(y1 / 2 + y2 / 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1) # 센터값을 원으로 표시

        inside_region1 = cv2.pointPolygonTest(region1, (cx, cy), False) # 센터값이 위쪽 영역에 속하는지 확인

        if inside_region1 > 0: # 위쪽 영역에 속한다면
            if ids in total_alt:
                first_out.add(ids)
                cv2.line(frame, (0,130), (640,280), (255,255,255), 3)
            # 위쪽 영역에 입장하면 id 추가
            total_ust.add(ids)

        inside_region2 = cv2.pointPolygonTest(region2, (cx, cy), False)
        if inside_region2 > 0:
            if ids in total_ust:
                cv2.line(frame, (0,130), (640,+280), (255,255,255), 3)
                first_in.add(ids)
            total_alt.add(ids)

    first_in_str = 'IN: ' + str(len(first_in))
    first_out_str = 'OUT: ' + str(len(first_out))

    # 배경색 설정
    frame[0:40, 0:120] = (102, 0, 153)
    frame[0:40, 510:640] = (102, 0, 153)

    cv2.putText(frame, first_in_str, (0, 30), font, 1, (255, 255, 255), 1)
    cv2.putText(frame, first_out_str, (510, 30), font, 1, (255, 255, 255), 1)
    # print('IN: ', len(first_in), 'OUT: ', len(first_out))

    # 원하는 경우 화면에 지역을 표시
    #cv2.polylines(frame,[region1],True,(255,0,0),2)
    #cv2.polylines(frame,[region2],True,(255,0,255),2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
