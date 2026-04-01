"""
摄像头配置
用于 IP Webcam 视频流连接
"""

# ==================== IP Webcam 配置 ====================
# 手机和电脑需要在同一 Wi-Fi 网络下

# 方法1：使用完整 URL（推荐）
# 从 IP Webcam App 复制完整 URL
# 例如：http://192.168.1.100:8080
WEBCAM_URL = "http://10.24.100.139:8080"

# 视频流 URL（自动添加 /video）
VIDEO_URL = f"{WEBCAM_URL}/video"

# 快照 URL
SNAPSHOT_URL = f"{WEBCAM_URL}/photo.jpg"

# ==================== 检测参数 ====================

# 检测置信度阈值
CONFIDENCE_THRESHOLD = 0.25

# 显示窗口大小
DISPLAY_SIZE = (1280, 720)

# 目标帧率
TARGET_FPS = 30

# ==================== 使用说明 ====================
#
# 1. 打开手机上的 IP Webcam App
# 2. 点击 "Start server"
# 3. 记下显示的完整 URL，例如：
#    http://192.168.1.100:8080
# 4. 将这个 URL 复制到上面的 WEBCAM_URL 变量中
#
# 注意：手机和电脑必须连接到同一个 Wi-Fi 网络
