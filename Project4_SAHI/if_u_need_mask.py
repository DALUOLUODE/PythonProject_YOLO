import os
import numpy as np
import cv2

# 设置路径
video_path = "../Datas/Videos/car_1.mp4"

def create_mask_interactive(video_path):
    # 读取视频第一帧
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("无法读取视频")

    height, width = frame.shape[:2]
    points = []  # 存储点击的点
    mask = np.zeros((height, width), dtype=np.uint8)
    temp_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # 画出点击的点
            cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1)
            # 如果有多个点，画线连接它们
            if len(points) > 1:
                cv2.line(temp_frame, points[-2], points[-1], (0, 255, 0), 2)
            # 如果是第一个点，显示提示
            if len(points) == 1:
                cv2.putText(temp_frame, "Click to add points, press 'C' to close polygon",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Create Mask - Click points, press C to close', temp_frame)

    # 创建窗口和鼠标回调
    window_name = 'Create Mask - Click points, press C to close'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
    cv2.resizeWindow(window_name, width, height)     # 设置窗口大小为视频分辨率
    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        # 按'c'关闭多边形
        if key == ord('c') and len(points) > 2:
            # 创建最终的mask
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

            # 可视化最终的mask
            mask_visualization = frame.copy()
            # 绘制半透明的mask
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))  # 绿色填充
            cv2.addWeighted(overlay, 0.3, mask_visualization, 0.7, 0, mask_visualization)
            # 绘制多边形边界
            cv2.polylines(mask_visualization, [pts], True, (0, 255, 0), 2)

            # 显示最终效果
            final_window = 'Final Mask'
            cv2.namedWindow(final_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(final_window, width, height)
            cv2.imshow(final_window, mask_visualization)
            # 保存可视化结果
            cv2.imwrite("../Project4_SAHI/mask_visualization.png", mask_visualization)
            cv2.imwrite("../Project4_SAHI/mask_binary.png", mask)
            break

        # 按'r'重置
        elif key == ord('r'):
            points = []
            temp_frame = frame.copy()
            cv2.imshow(window_name, temp_frame)

        # 按'q'退出
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return mask, points


# 测试mask效果的函数
def test_mask_on_video(video_path, mask, points):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        return
    
    height, width = first_frame.shape[:2]
    window_name = 'Original vs Masked (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width * 2, height)  # 宽度*2因为是并排显示

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 创建mask可视化
        mask_visualization = frame.copy()

        # 绘制半透明mask区域
        overlay = frame.copy()
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))  # 绿色填充
        cv2.addWeighted(overlay, 0.3, mask_visualization, 0.7, 0, mask_visualization)

        # 绘制ROI边界
        cv2.polylines(mask_visualization, [pts], True, (0, 255, 0), 2)

        # 显示原始帧和mask效果的对比
        combined = np.hstack((frame, mask_visualization))
        cv2.imshow(window_name, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 主程序
if __name__ == "__main__":
    print("开始创建mask...")
    print("使用说明：")
    print("1. 点击视频帧添加ROI点")
    print("2. 按'C'完成多边形绘制")
    print("3. 按'R'重置当前绘制")
    print("4. 按'Q'退出程序")

    # 创建并可视化mask
    mask, points = create_mask_interactive(video_path)

    # 测试mask效果
    print("\n是否要测试mask效果？(y/n)")
    choice = input().lower()
    if choice == 'y':
        print("正在测试mask效果...")
        print("按'Q'退出测试")
        test_mask_on_video(video_path, mask, points)

    # 保存mask点坐标供后续使用
    np.save("../Project4_SAHI/mask_points.npy", points)
    print("\nMask创建完成！")
    print(f"Mask点坐标已保存至: ../Project4_SAHI/mask_points.npy")
    print(f"Mask可视化结果已保存至: ../Project4_SAHI/mask_visualization.png")
    print(f"Mask二值图已保存至: ../Project4_SAHI/mask_binary.png")