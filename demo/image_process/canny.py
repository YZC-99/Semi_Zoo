
import cv2

# 读取图像
image_path = './G-1-L.jpg'
image = cv2.imread(image_path)

# 转换图像颜色空间
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# 对 Lab 颜色空间的 a 和 b 通道应用 Mean Shift
# 调整 sp（空间窗口大小）和 sr（颜色窗口大小）参数以满足你的需求
shifted = cv2.pyrMeanShiftFiltering(image_lab, sp=20, sr=50)

# 将结果转换回 BGR 颜色空间
result = cv2.cvtColor(shifted, cv2.COLOR_Lab2BGR)

# 保存 Mean Shift 后的图像
output_path = './mean_shift_result.jpg'
cv2.imwrite(output_path, result)
