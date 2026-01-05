import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import torch
from torch import nn
import mediapipe as mp
from PIL import Image

# 设置页面配置
st.set_page_config(
    page_title="手势识别应用",
    page_icon="👋",
    layout="wide"
)

# 初始化MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """处理图像并检测关键点"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """提取关键点数据"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

# 定义LSTM模型类
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.output_layer(x)
        return x

# 手势类别列表
gestures = np.array(['polis','nasi','abang','apa','hari','ribut','pukul','beli','emak','perlahan'])

# 加载模型
model_path = "model.pth"
input_size = 258
hidden_size = 64
num_classes = len(gestures)

try:
    model = torch.load(model_path, weights_only=False)
    model.eval()
    except Exception as e:
    st.error(f"加载模型时出错: {str(e)}")
    print(torch.cuda.is_available)
    model = CustomLSTM(input_size, hidden_size, num_classes)
    model.eval()

# 主界面
st.title("👋 手势识别应用")

# 第一个文本框：上传提示
st.markdown("### 请上传手语视频来获取预测")

# 上传视频
uploaded_file = st.file_uploader("选择视频文件", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    # 读取视频基本信息
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # 第二个文本框：显示视频信息
    st.markdown("### 视频信息")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**文件名:** {uploaded_file.name}")
    with col2:
        st.info(f"**时长:** {duration:.2f}秒")
    
    # 开始处理按钮
    if st.button("开始预测", type="primary"):
        with st.spinner("正在处理视频..."):
            # 初始化变量
            cap = cv2.VideoCapture(video_path)
            sequence = []
            predictions_history = []
            frame_count = 0
            
            # 进度条
            progress_bar = st.progress(0)
            
            # 创建MediaPipe模型
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as holistic:
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 更新进度
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(min(progress, 1.0))
                    
                    # 检测关键点
                    _, results = mediapipe_detection(frame, holistic)
                    
                    # 提取关键点并添加到序列
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]  # 保持最近30帧
                    
                    # 如果有手部关键点并且序列足够长，进行预测
                    if (results.left_hand_landmarks or results.right_hand_landmarks) and len(sequence) == 30:
                        try:
                            # 转换为模型输入格式
                            input_data = torch.tensor(
                                np.expand_dims(sequence, axis=0), 
                                dtype=torch.float32
                            )
                            
                            # 进行预测
                            with torch.no_grad():
                                res = model(input_data)
                            
                            # 获取预测结果
                            probabilities = torch.softmax(res, dim=1)
                            max_prob, max_idx = torch.max(probabilities, dim=1)
                            
                            pred_class = gestures[max_idx.item()]
                            confidence = max_prob.item() * 100
                            
                            # 存储预测结果
                            predictions_history.append({
                                'class': pred_class,
                                'confidence': confidence
                            })
                            
                        except Exception as e:
                            pass
            
            # 释放资源
            cap.release()
            progress_bar.progress(1.0)
            
            # 第三个文本框：显示预测结果
            st.markdown("### 预测结果")
            
            if predictions_history:
                # 获取最后30帧的预测结果
                last_30_predictions = predictions_history[-30:] if len(predictions_history) >= 30 else predictions_history
                
                # 统计每个类别的出现次数
                class_counts = {}
                for pred in last_30_predictions:
                    pred_class = pred['class']
                    class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                
                # 找到出现最多的类别
                if class_counts:
                    most_common_class = max(class_counts.items(), key=lambda x: x[1])
                    final_prediction = most_common_class[0]
                    confidence_score = (most_common_class[1] / len(last_30_predictions)) * 100
                    
                    st.success(f"**预测结果:** {final_prediction}")
                    st.info(f"**置信度:** {confidence_score:.1f}%")
                else:
                    st.warning("未能确定分类结果")
            else:
                st.warning("未检测到手部动作")
            
            # 清理临时文件
            try:
                os.unlink(video_path)
            except:
                pass

# 添加简单的说明
st.markdown("---")
st.markdown("*上传包含手语的MP4视频文件，系统将自动识别手势动作*")
