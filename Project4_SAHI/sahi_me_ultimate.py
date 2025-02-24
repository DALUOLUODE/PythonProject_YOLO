import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
import torch
from tqdm import tqdm
import time

yolo11n_model_path = "../Yolo_Weights/yolo11n.pt"

# 初始化模型
model = AutoDetectionModel.from_pretrained(
    model_type="yolo11",  # 使用YOLOv8模型
    model_path=yolo11n_model_path,  # 使用预训练模型
    confidence_threshold=0.3,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 定义目标类别和对应的颜色
TARGET_CLASSES = {
    'car': (255, 0, 0),  # 蓝色
    'truck': (0, 255, 0),  # 绿色
    'bus': (0, 0, 255),  # 红色
    'person': (0, 255, 255)
}


def process_frame(frame):
    """处理单帧图像"""
    # 将frame转换为PIL Image
    frame_pil = read_image_as_pil(frame)

    # 使用SAHI进行切片预测
    prediction_result = get_sliced_prediction(
        image=frame_pil,
        detection_model=model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # 在原始帧上绘制预测结果
    for pred in prediction_result.object_prediction_list:
        # 只处理目标类别
        if pred.category.name in TARGET_CLASSES:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
            # 获取类别名称和置信度
            label = pred.category.name
            score = pred.score.value
            # 获取对应的颜色
            color = TARGET_CLASSES[label]

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # 添加标签
            label_text = f"{label} {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def main():
    # 打开视频文件或摄像头
    video_path = "../Datas/Videos/car_1.mp4"  # 替换为你的视频路径
    cap = cv2.VideoCapture(video_path)

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建视频写入器
    output_path = "../Prediction_Datas/out_put.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("开始处理视频...")

    # 创建进度条
    pbar = tqdm(total=total_frames, desc="处理视频进度")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 处理帧
        processed_frame = process_frame(frame)

        # 显示结果
        cv2.imshow('Detection Result', processed_frame)

        # 保存处理后的帧
        out.write(processed_frame)

        # 更新进度条
        frame_count += 1
        pbar.update(1)

        # 显示处理速度
        if frame_count % 10 == 0:  # 每10帧更新一次进度信息
            pbar.set_postfix({
                'FPS': f'{fps:.1f}',
                '已处理': f'{frame_count}/{total_frames}'
            })

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭进度条
    pbar.close()

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成，结果已保存至 {output_path}")


if __name__ == "__main__":
    main()