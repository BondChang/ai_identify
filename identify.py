import cv2
import mediapipe as mp
import numpy as np
import json
import os
import sys
import socket
import threading
from queue import Queue

# 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def load_config(config_path='config.json'):
    """
    读取配置文件
    """
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_video_stream(video_url):
    """
    获取视频流
    """
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(f"Unable to open video stream: {video_url}")
        return None
    return cap


def detect_pose(frame, pose):
    """
    使用MediaPipe检测人体姿势
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    return results


def calculate_angle(landmarks, a, b, c):
    """
    计算三个关键点形成的角度
    """
    a = np.array(landmarks[a])
    b = np.array(landmarks[b])
    c = np.array(landmarks[c])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def is_preparing_to_shoot(landmarks, config):
    """
    判断是否处于预备投球动作
    """
    # 获取膝盖角度阈值
    left_knee_threshold = config['thresholds']['left_knee_threshold']
    right_knee_threshold = config['thresholds']['right_knee_threshold']

    # 检查双腿是否弯曲（膝盖角度）
    left_knee = calculate_angle(landmarks, 'left_hip', 'left_knee', 'left_ankle')
    right_knee = calculate_angle(landmarks, 'right_hip', 'right_knee', 'right_ankle')
    legs_bent = left_knee < left_knee_threshold and right_knee < right_knee_threshold  # 根据配置文件调整阈值
    # print(f"Left knee angle: {left_knee}, Right knee angle: {right_knee}")
    # 获取手对齐阈值
    hands_aligned_threshold = config['thresholds']['hands_aligned_threshold']

    # 检查手是否在肩部左右平齐，并计算误差
    left_wrist = landmarks.get('left_wrist', (0, 0, 0))
    right_wrist = landmarks.get('right_wrist', (0, 0, 0))
    left_shoulder = landmarks.get('left_shoulder', (0, 0, 0))
    right_shoulder = landmarks.get('right_shoulder', (0, 0, 0))
    hands_aligned = True
    error = 0.0  # 初始化误差值
    if left_shoulder != (0, 0, 0) and left_wrist != (0, 0, 0):
        error += abs(left_wrist[1] - left_shoulder[1])
        hands_aligned &= abs(left_wrist[1] - left_shoulder[1]) < hands_aligned_threshold  # 使用配置文件中的阈值
    if right_shoulder != (0, 0, 0) and right_wrist != (0, 0, 0):
        error += abs(right_wrist[1] - right_shoulder[1])
        hands_aligned &= abs(right_wrist[1] - right_shoulder[1]) < hands_aligned_threshold

    return legs_bent and hands_aligned, error


def is_shooting(landmarks, previous_shoot_y, config):
    """
    判断是否处于投球动作
    """
    # 获取投球动作阈值
    shoot_y_threshold = config['thresholds']['shoot_y_threshold']

    # 双手在最高点
    left_wrist = landmarks.get('left_wrist', (0, 0, 0))
    right_wrist = landmarks.get('right_wrist', (0, 0, 0))
    if left_wrist == (0, 0, 0) or right_wrist == (0, 0, 0):
        return False, previous_shoot_y
    left_wrist_y = left_wrist[1]
    right_wrist_y = right_wrist[1]
    current_shoot_y = max(left_wrist_y, right_wrist_y)

    # 检查是否达到新的最高点
    shooting = current_shoot_y < previous_shoot_y - shoot_y_threshold  # 使用配置文件中的阈值
    return shooting, current_shoot_y if shooting else previous_shoot_y


def is_palm_up(landmarks, config):
    """
    判断手掌是否朝上（手掌朝向天空）
    通过比较手指关键点与鼻子的横坐标距离
    """
    # 获取手掌朝上阈值（最大允许的横坐标差异）
    palm_up_threshold = config['thresholds']['palm_up_threshold']

    # 获取鼻子的横坐标
    nose = landmarks.get('nose', (0, 0, 0))

    if nose == (0, 0, 0):
        return False  # 无法获取鼻子位置

    nose_x = nose[0]

    # 定义需要检查的手指关键点
    fingers = ['left_index', 'left_pinky', 'right_index', 'right_pinky']
    wrists = ['left_wrist', 'right_wrist']
    # 检查每个手指是否比手腕更接近鼻子的横坐标
    for finger, wrist in zip(fingers, wrists):
        finger_landmark = landmarks.get(finger, (0, 0, 0))
        wrist_landmark = landmarks.get(wrist, (0, 0, 0))
        if finger_landmark == (0, 0, 0) or wrist_landmark == (0, 0, 0):
            return False  # 缺失关键点，无法判断

        finger_distance = abs(finger_landmark[0] - nose_x)
        wrist_distance = abs(wrist_landmark[0] - nose_x)

        if finger_distance >= wrist_distance - palm_up_threshold:
            return False  # 手指没有比手腕更接近鼻子

    return True  # 所有手指均比手腕更接近鼻子


def save_frame(frame, frame_number, action_type, output_dir):
    """
    保存指定动作的帧图像
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{action_type}_frame_{frame_number}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, frame)


def save_json(data, json_path):
    """
    保存动作数据到JSON文件
    """
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def udp_listener(ip, port, queue, stop_event):
    """
    UDP监听器，接收视频地址并放入队列
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    print(f"UDP listener started, listening on {ip}:{port}")
    sock.settimeout(1.0)  # 设置超时以便定期检查停止事件
    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(4096)
            video_url = data.decode('utf-8').strip()
            print(f"Received video URL: {video_url} from {addr}")
            queue.put((video_url, addr))
        except socket.timeout:
            continue
        except Exception as e:
            print(f"UDP listener error: {e}")
    sock.close()
    print("UDP listener stopped")


def send_udp_message(message, addr):
    """
    通过UDP发送消息
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message.encode('utf-8'), addr)
    sock.close()


def process_video(video_url, addr, config):
    """
    处理单个视频地址
    """
    cap = get_video_stream(video_url)
    if cap is None:
        return

    pose = mp_pose.Pose(min_detection_confidence=config['pose']['min_detection_confidence'],
                        min_tracking_confidence=config['pose']['min_tracking_confidence'])

    frame_count = 0
    shoot_frame = 0
    action_data = {
        "pre_shoot_frame": None,
        "shoot_frame": None
    }
    pre_shoot_min_error = float('inf')
    previous_shoot_y = float('inf')
    shoot_min_y = float('inf')
    # 创建一个新的字典用于存储所有满足preparing条件的数据
    pre_shoot_frames_data = {
        "frames": []  # 存储所有符合条件的帧数据
    }
    # 获取视频目录
    video_dir = os.path.dirname(video_url)
    if not video_dir:
        video_dir = config['output_dir']
    output_dir = video_dir if video_dir else config['output_dir']

    print(f"Start processing video: {video_url}")
    last_frame = None  # 用于存储最后一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Cannot read frame or end of video: {video_url}")
            break

        frame_count += 1
        results_pose = detect_pose(frame, pose)

        landmarks = {}
        if results_pose.pose_landmarks:
            for lm in mp_pose.PoseLandmark:
                lm_name = lm.name.lower()
                landmark = results_pose.pose_landmarks.landmark[lm]
                # 保存x, y, z坐标
                landmarks[lm_name] = (landmark.x, landmark.y, landmark.z)

        # 判断预备投球动作
        if landmarks:
            preparing, error = is_preparing_to_shoot(landmarks, config)
            # print(f"Frame {frame_count}: Preparing to shoot: {preparing}, Error: {error}")
            if preparing:
                if error < pre_shoot_min_error:
                    pre_shoot_min_error = error
                    pre_shoot_frames_data["frames"].append({
                        "frame_number": frame_count,
                        "error": error
                    })
                    action_data["pre_shoot_frame"] = frame_count
                    if config['save_shoot_images']:
                        save_frame(frame, frame_count, "pre_shoot", output_dir)

            # 判断投球动作
            shooting, current_shoot_y = is_shooting(landmarks, previous_shoot_y, config)
            if shooting and frame_count>1:
                last_frame = frame
                shoot_frame = frame_count
                # save_frame(frame, frame_count, "shoot", output_dir)
                # 判断手掌是否朝上
                palms_up = is_palm_up(landmarks, config)
                if palms_up and current_shoot_y < shoot_min_y:
                    shoot_min_y = current_shoot_y
                    action_data["shoot_frame"] = frame_count
                    if config['save_shoot_images']:
                        save_frame(frame, frame_count, "shoot", output_dir)

            previous_shoot_y = min(previous_shoot_y, current_shoot_y)

            # 绘制关键点
            if config['draw_landmarks']:
                mp_drawing.draw_landmarks(
                    frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

        # 显示视频
        if config['display_video']:
            cv2.imshow('Pose Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit command received, stopping video processing.")
                break

    # 如果没有检测到投篮帧，保存最后的投篮帧
    if action_data["shoot_frame"] is None:
        action_data["shoot_frame"] = shoot_frame
        # 使用最后一帧保存投篮图片
        if config['save_shoot_images'] and last_frame is not None:
            save_frame(last_frame, frame_count, "shoot", output_dir)

    if action_data["pre_shoot_frame"] is not None and action_data["shoot_frame"] is not None:
        if action_data["pre_shoot_frame"] >= action_data["shoot_frame"]:
            # 从 pre_shoot_frames_data 中找到小于 shoot_frame 的最大值
            valid_pre_shoot_frames = [item for item in pre_shoot_frames_data["frames"] if
                                      item["frame_number"] < action_data["shoot_frame"]]
            if valid_pre_shoot_frames:
                # 找到小于 shoot_frame 的最大 pre_shoot_frame
                action_data["pre_shoot_frame"] = max(valid_pre_shoot_frames, key=lambda x: x["frame_number"])[
                    "frame_number"]
                # print(f"Updated pre_shoot_frame to {action_data['pre_shoot_frame']}")

    # 释放资源
    cap.release()
    pose.close()
    if config['display_video']:
        cv2.destroyAllWindows()

    # 保存JSON数据
    video_name = os.path.splitext(os.path.basename(video_url))[0]
    json_filename = f"{video_name}_ai_identify.json"
    json_path = os.path.join(output_dir, json_filename)
    save_json(action_data, json_path)
    print(f"Video processing completed, data saved to {json_path}")

    # 发送处理成功消息
    success_message = "Video processing completed successfully."
    send_udp_message(success_message, addr)


def main(config):
    """
    主函数，集成所有功能
    """
    video_queue = Queue()
    stop_event = threading.Event()

    # 启动UDP监听线程
    listener_thread = threading.Thread(target=udp_listener,
                                       args=(config['udp_ip'], config['udp_port'], video_queue, stop_event))
    listener_thread.start()

    try:
        while True:
            try:
                video_url, addr = video_queue.get(timeout=1)  # 等待1秒获取视频地址
                if video_url:
                    process_video(video_url, addr, config)
            except:
                continue
    except KeyboardInterrupt:
        print("Program interrupted by user, exiting...")
    finally:
        stop_event.set()
        listener_thread.join()
        print("Program exited.")


if __name__ == "__main__":
    # 加载配置文件
    config = load_config('config.json')

    main(config)
