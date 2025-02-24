import os
from sahi import AutoDetectionModel
from sahi.predict import predict
import numpy as np
import cv2

# 设置路径
yolo11n_model_path = "../Yolo_Weights/yolo11n.pt"
video_path = "../Datas/Videos/car_1.mp4"
mask_path = '../Project4_SAHI/mask_binary.png'  

# 读取mask
mask=cv2.imread(mask_path,0)
if mask is None:
    raise ValueError(f"Error: 无法读取mask文件 {mask_path}")

# 创建临时目录用于存储处理后的帧
temp_dir = "../Prediction_Datas/temp_masked_frames"
os.makedirs(temp_dir, exist_ok=True)

# 预处理视频，应用mask
def preprocess_video(video_path, mask, temp_dir):
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: 无法打开视频文件 {video_path}")
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 调整mask大小以匹配视频分辨率
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height))

    # 创建临时视频文件
    # temp_video_path = os.path.join(temp_dir, '/masked_video.mp4')
    temp_video_path = temp_dir+'/masked_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 应用mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        out.write(masked_frame)
        frame_count += 1

    # 释放资源
    cap.release()
    out.release()
    print(f"预处理完成，共处理 {frame_count} 帧")
    return temp_video_path,fps
    

# 初始化模型
model = AutoDetectionModel.from_pretrained(
    model_path=yolo11n_model_path,
    model_type="yolo11",
    confidence_threshold=0.5,
    device="cuda"
)

# 预处理视频
print("开始预处理视频...")
preprocessed_video_path,fps = preprocess_video(video_path, mask, temp_dir)
# 定义需要检测的类别和对应的颜色

# print("开始进行目标检测...")
# # 调用predict函数进行切片预测
# predict(
#     detection_model=model,
#     model_type="yolo11",
#     model_path=yolo11n_model_path,
#     source=preprocessed_video_path,
#     model_confidence_threshold=0.5,
#     model_device="cuda",
#
#     # 切片参数
#     slice_height=256,
#     slice_width=256,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
#
#     # 视频相关参数
#     project="../Prediction_Datas",
#     name="vehicle_detection",
#     view_video=True,  # 实时显示检测结果
#
#     # 可视化参数
#     visual_bbox_thickness=2,
#     visual_text_size=1.0,
#     visual_text_thickness=2,
# )
#
# # 清理临时文件
# import shutil
# if os.path.exists(temp_dir):
#     shutil.rmtree(temp_dir)
# print("处理完成，临时文件已清理")
#
# # 显示处理后的视频
# output_path = "../Prediction_Datas/vehicle_detection/masked_video.mp4"
# if os.path.exists(output_path):
#     cap = cv2.VideoCapture(output_path)
#
#     # 创建可调整大小的窗口
#     cv2.namedWindow('Vehicle Detection', cv2.WINDOW_NORMAL)
#
#     # 获取视频尺寸并设置窗口大小
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cv2.resizeWindow('Vehicle Detection', width, height)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 显示mask区域
#         mask = cv2.imread(mask_path, 0)  # 重新读取mask用于显示
#         if mask.shape[:2] != (height, width):
#             mask = cv2.resize(mask, (width, height))
#
#         mask_overlay = frame.copy()
#         mask_visualization = cv2.bitwise_and(mask_overlay, mask_overlay, mask=mask)
#         cv2.addWeighted(mask_visualization, 0.3, frame, 0.7, 0, frame)
#
#         cv2.imshow('Vehicle Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
# else:
#     print("无法找到输出视频文件")