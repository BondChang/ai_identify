from cx_Freeze import setup, Executable
import sys

# 需要打包的主脚本文件
script = "identify.py"  # Python 脚本文件

# 比如MediaPipe, OpenCV等可能需要的库
build_exe_options = {
    "packages": ["socket", "configparser", "cv2", "mediapipe", "json", "os"],
    "include_files": []  # 如果有额外的文件需要打包（例如配置文件、模型文件等）
}


setup(
    name="ai_identify",
    version="1.0",
    description="A program to process video and extract pose keypoints",
    # options={"build_exe": build_exe_options},
    executables=[Executable(script, base=None)],
)
