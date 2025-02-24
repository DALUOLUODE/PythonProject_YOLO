import os
from sahi.utils.ultralytics import download_yolo11n_model, download_yolo11n_seg_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image,visualize_object_predictions
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
from sahi.utils.cv import read_image_as_pil
import cv2

yolo11n_model_path = "../Yolo_Weights/yolo11n.pt"


model = AutoDetectionModel.from_pretrained(
    model_type="yolo11",  # 使用YOLOv8模型
    model_path=yolo11n_model_path,  # 使用预训练模型
    confidence_threshold=0.3,
    device="cpu"
    # if torch.cuda.is_available() else "cpu"
)

data_path='../Datas/Images/car.png'
 # 处理帧
processed_frame = get_sliced_prediction(data_path,
                                        model,
                                        slice_height = 512,
                                        slice_width = 512,
                                        overlap_height_ratio = 0.2,
                                        overlap_width_ratio = 0.2,)

processed_frame.export_visuals(export_dir='../Prediction_Datas',file_name="test")
final=cv2.imread('../Prediction_Datas/test.png')

# 读取图片后获取尺寸
h, w = final.shape[:2]  # 获取图像的高度和宽度（顺序为h,w）
# 显示结果（允许调整窗口大小）
cv2.namedWindow('Detection Result', cv2.WINDOW_NORMAL)  # 设置为可调整模式
cv2.resizeWindow('Detection Result', w, h)          # 设置初始窗口大小（可选）

# 显示结果
cv2.imshow('Detection Result', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
