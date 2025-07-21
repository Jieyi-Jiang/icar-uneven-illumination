import cv2
import time

cap = cv2.VideoCapture(0)  # 0 是默认摄像头，如果有多个摄像头，可以选择不同的编号

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    print(f"当前帧率: {fps:.2f} FPS")
    print(f"size of frame: {frame.shape[1]}x{frame.shape[0]}")

    # 如果你想实时显示画面，可以取消注释以下行
    # cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
