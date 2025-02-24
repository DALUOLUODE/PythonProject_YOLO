import os
from sahi.utils.ultralytics import download_yolo11n_model, download_yolo11n_seg_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image,visualize_object_predictions
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
import cv2

# 指定自定义下载路径
custom_model_path = "../Yolo_Weights"  # 您可以修改为任意想要的路径
# os.makedirs(custom_model_path, exist_ok=True)  # 确保目录存在

# 下载模型到指定路径
yolo11n_model_path = custom_model_path + "/yolo11n.pt"
# download_yolo11n_model(yolo11n_model_path)
# print(f"模型下载位置: {yolo11n_model_path}")

# 数据存储路径
data_path = "../Datas"
# os.makedirs(data_path, exist_ok=True)  # 确保目录存在
# 预测数据保存路径
prediction_data_path = "../Prediction_Datas"
# os.makedirs(prediction_data_path, exist_ok=True)  # 确保目录存在

# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg',
#                   data_path + "Images/small-vehicles1.jpeg")
# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png',
#                   data_path + "Images/terrain2.png")

# 初始化模型
model = AutoDetectionModel.from_pretrained(
    model_path=yolo11n_model_path,  # 指定模型路径，只包含权重数据  
    model_type="yolo11", # 指定模型类型，告诉sahi使用yolo11模型
    confidence_threshold=0.5, # 置信度阈值
    device="cuda",# 指定设备
)

# 获取预测结果
# prediction_without_slicing = get_prediction(data_path + "Images/small-vehicles1.jpeg",model)
prediction_with_slicing = get_sliced_prediction(data_path + "/Images/car.png",
                                                model,
                                                slice_height = 512,
                                                slice_width = 512,
                                                overlap_height_ratio = 0.2,
                                                overlap_width_ratio = 0.2)

# 导出到数据目录
# prediction_without_slicing.export_visuals(export_dir=prediction_data_path,file_name="image_with_predictions_without_slicing")
prediction_with_slicing.export_visuals(export_dir=prediction_data_path,file_name="image_with_predictions_with_slicing")

# 显示预测图像
# image_with_predictions_without_slicing = cv2.imread(prediction_data_path + "/image_with_predictions_without_slicing.png")
image_with_predictions_with_slicing = cv2.imread(prediction_data_path + "/image_with_predictions_with_slicing.png")
# cv2.imshow("Image with Predictions without slicing", image_with_predictions_without_slicing)
cv2.imshow("Image with Predictions with slicing", image_with_predictions_with_slicing)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 或者用Image显示图像,jupyter中用
# Image(data_path + "/image_with_predictions_without_slicing.png")
# Image(data_path + "/image_with_predictions_with_slicing.png")

# 显示识别列表
# print(prediction_without_slicing.object_prediction_list)
# print(prediction_with_slicing.object_prediction_list)

# 显示识别列表的第一个
# print(prediction_without_slicing.object_prediction_list[0])
# print(prediction_with_slicing.object_prediction_list[0])

# 将切片预测转换为coco标注格式，用于数据集制作，并输出前三行
# coco_format = prediction_with_slicing.to_coco_annotations()
# print(f"coco_format: {coco_format[:3]}")

# 将切片预测转换为coco（更规范）预测格式，用于模型评估，并输出前三行
# coco_format = prediction_with_slicing.to_coco_predictions(image_id=1)
# image_id=1 表示输出的检测结果来自同一张图片
# print(f"coco_format: {coco_format[:3]}")

# 将切片预测转换为imantics标注格式，用于数据集制作，并输出前三行
# imantics_format = prediction_with_slicing.to_imantics_annotations()
# 下面输出的是imantics的annotation格式
# print(f"imantics_format: {imantics_format[:3]}")

# 批量预测batch_prediction
# predict(
#     detection_model=model,
#     model_type="yolo11",
#     model_path=yolo11n_model_path,
#     source=data_path,
#     model_confidence_threshold=0.5,
#     model_device="cuda",
#     slice_height=256,
#     slice_width=256,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
#     project=prediction_data_path,
#     name="exp")












