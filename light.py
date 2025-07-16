import cv2
import numpy as np
import matplotlib.pyplot as plt

def cal_the_variance(img_rgb):
    img_shape = img_rgb[:,:,0].shape
    img_var = np.zeros(img_shape, dtype=np.float32)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.int16)
    img_ch0 = img_rgb[:,:,0].astype(np.int16)
    img_ch1 = img_rgb[:,:,1].astype(np.int16)
    img_ch2 = img_rgb[:,:,2].astype(np.int16)
    img_var = (np.abs(img_ch0 - gray) + np.abs(img_ch1 - gray) + np.abs(img_ch2 - gray))
    img_var = np.divide(img_var, gray+1)
    img_var = img_var * 200
    img_var = np.clip(img_var, 0, 255)
    img_var = img_var.astype(np.uint8)
    # ret = img_var.astype(np.uint8) * 10
    # print(ret)
    return img_var

def flood_fill(image, mid_point=(160, 230)):
    # 创建掩码图像
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img_shape = img_rgb[:,:,0].shape
    mask = np.zeros((img_shape[0]+2, img_shape[1]+2), dtype=np.uint8)  # 掩码比图像大 2 像素
    # print(mid_point)
    height_left = mid_point[1]-10
    if height_left < 0: height_left = 0
    height_right = mid_point[1]+5
    if height_right > img_shape[1]: height_right = img_shape[0]-1
    # width_top = mid_point[0]-10
    # if width_top < 0: width_top = 0
    # width_bottom = mid_point[0]+10
    # if width_bottom > img_shape[0]: width_bottom = img_shape[0]
    
    img_rgb[mid_point[1]-10:mid_point[1]+5, mid_point[0]-10:mid_point[0]+10, :] = 210  # 设置掩码区域

    # 设置种子点和填充颜色
    seedPoint = mid_point
    newVal = (0, 0, 255)  # 红色

    # 使用 floodFill 填充
    cv2.floodFill(img_rgb, mask, seedPoint, newVal, loDiff=(50, 50, 50), upDiff=(50, 50, 50))
    filled_area = mask[1:-1, 1:-1]
    # ret_img_rgb = cv2.cvtColor(filled_area*255, cv2.COLOR_GRAY2BGR)
    ret_img_gray = filled_area * 255
    return ret_img_gray


def search_the_mid_point(image, mid_point=(160, 230)):
    img_padded = np.zeros((image.shape[0]+2, image.shape[1]+2), dtype=np.uint8)
    img_padded[1:-1, 1:-1] = image
    left_edge = []
    right_edge = []
    k = 0
    for i in range(230, 240):
        k = 0
        for j in range(mid_point[0], 321):
            if img_padded[i, j] <= 20:
                k += 1
            if img_padded[i, j] > 20:
                k = 0 
            if k >= 4:
                left_edge.append(j)
                break
        if k < 4:
            left_edge.append(j)
    for i in range(230, 240):
        k = 0
        for j in range(mid_point[0], 0, -1):
            if img_padded[i, j] <= 20:
                k += 1
            if img_padded[i, j] > 20:
                k = 0 
            if k >= 4:
                right_edge.append(j)
                break
        if k < 4:
            right_edge.append(j)
    left_edge = np.array(left_edge)
    right_edge = np.array(right_edge)
    mid_edge = (left_edge + right_edge) / 2
    mid_point_width = np.int16(np.mean(mid_edge))
    mid_point = (mid_point_width, 230)
    return mid_point

def process_frame_seg(frame, mid_point=(160, 230)):
    image = cv2.resize(frame, (320, 240))  # 缩小图像以加快处理速度
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary = cv2.adaptiveThreshold(
        image_gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=71,                      # 邻域大小（建议奇数）
        C=20                                # 调整偏移值
    )
    
    # 形态学滤波
    filted = cv2.morphologyEx(binary, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))
    filted = cv2.morphologyEx(filted, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    img_merge = np.zeros(image_gray.shape, dtype=np.uint16)
    mid_point = search_the_mid_point(filted, mid_point)
    image_var = cal_the_variance(image_rgb)
    img_merge = filted/2 + (255-image_var)/2
    img_merge = cv2.morphologyEx(img_merge, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))
    img_merge = cv2.morphologyEx(img_merge, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # 限制到255
    img_merge = np.clip(img_merge, 0, 255)
    img_merge = img_merge.astype(np.uint8)
    flood_area = flood_fill(img_merge, mid_point)
    flood_area = cv2.morphologyEx(flood_area, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    flood_area = cv2.morphologyEx(flood_area, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    frame = cv2.cvtColor(flood_area, cv2.COLOR_GRAY2BGR)  # 转回 BGR 格式
    return frame, mid_point


def process_frame_var(frame, var_weight=0.7, sturtn_weight=0.3):
    image = cv2.resize(frame, (320, 240))  # 缩小图像以加快处理速度
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_var = cal_the_variance(image_rgb)
    # image_add = (image_var * 2 + (255-image_hsv[:, :, 1]) * 1) // 3
    var_weight = var_weight / (var_weight+sturtn_weight)
    sturtn_weight = sturtn_weight / (var_weight+sturtn_weight)
    image_add = cv2.addWeighted(255-image_var, var_weight, 255-image_hsv[:, :, 1], sturtn_weight, 0)
    # image_add = image_var
    binary_frame = cv2.threshold(image_add, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    image_new_rgb = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)  # 转回 BGR 格式
    frame = image_new_rgb
    return frame

def process_frame_gray(frame):
    image_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    binary_frame = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    image_new_rgb = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)  # 转回 BGR 格式
    frame = image_new_rgb
    return frame
    
if __name__ == '__main__':
        
    for i in range(8):
    # for i in range(1):
        # 输入文件路径
        file_name = f"exp{i}"
        # file_name = f"sample"
        input_file = f'./data/{file_name}.mp4'
        # method = 'seg' # 'seg', 'var', 'gray' 
        # method = 'seg' # 'seg', 'var', 'gray' 
        method = 'gray' # 'seg', 'var', 'gray' 
        # input_file = f'./data/{file_name}.avi'
        print(input_file)
        output_file = f'./result/综合/{file_name}-{method}.mp4'

        # 打开视频文件
        cap = cv2.VideoCapture(input_file)

        # 获取视频属性
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码

        # 创建 VideoWriter 对象
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width * 2, frame_height))  # 宽度加倍
        mid_point = (160, 230)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            if method == 'seg':
                # print('Segmentation')
                processed_frame, mid_point = process_frame_seg(frame, mid_point)
            elif method == 'var':
                # print('Variance')
                processed_frame = process_frame_var(frame, 0.7, 0.7)
            elif  method == 'gray':
                # print('Gray')
                processed_frame = process_frame_gray(frame)
            else:
                # print('Gray')
                processed_frame = process_frame_gray(frame)
            
            # 将原始帧和处理后的帧拼接在一起
            combined_frame = cv2.hconcat([frame, processed_frame])
            
            # 写入拼接后的帧
            out.write(combined_frame)
        print("处理完成")
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()