import cv2
import os

# 视频路径（支持本地路径或摄像头输入）
# 输入文件路径
file_name = "sample"
input_file = f'./data/{file_name}.avi'
video_path = input_file
print(f"视频文件路径: {video_path}")
# 输出帧图片保存路径
output_dir = f'./result/frames_output/{file_name}'
os.makedirs(output_dir, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查是否成功打开
if not cap.isOpened():
    print("无法打开视频！")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取完毕

    # 保存当前帧为图像文件
    frame_name = f"{output_dir}/frame_{frame_count:04d}.png"
    cv2.imwrite(frame_name, frame)
    
    print(f"保存: {frame_name}")
    frame_count += 1

cap.release()
print(f"总共提取帧数: {frame_count}")
