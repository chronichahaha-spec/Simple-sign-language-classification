import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import torch
from torch import nn
import mediapipe as mp
from PIL import Image
import time

# 设置页面配置
st.set_page_config(
    page_title="手势识别应用",
    page_icon="👋",
    layout="wide"
)

# 标题
st.title("🎬 手势识别视频分析应用")
st.markdown("上传MP4视频文件，系统将检测并识别手势动作")

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

def draw_styled_landmarks(image, results):
    """绘制关键点和连接线"""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1)
        )

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


model_path = "model.pth"
    

# 手势类别列表
gestures = np.array(['polis','nasi','abang','apa','hari','ribut','pukul','beli','emak','perlahan'])

# 主内容区
uploaded_file = st.file_uploader("📁 上传MP4视频文件", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 文件信息")
        file_name = uploaded_file.name
        file_size = uploaded_file.size / (1024 * 1024)  # 转换为MB
        
        st.write(f"**文件名:** {file_name}")
        st.write(f"**文件大小:** {file_size:.2f} MB")
        st.write(f"**检测到的手势类别数:** {len(gestures)}")
        
        # 显示视频信息
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        # 读取视频基本信息
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        st.write(f"**视频分辨率:** {width} x {height}")
        st.write(f"**帧率:** {fps} FPS")
        st.write(f"**总帧数:** {total_frames}")
        st.write(f"**时长:** {duration:.2f} 秒")
        
        cap.release()
    
    # 开始处理按钮
    if st.button("🚀 开始处理视频", type="primary"):
        with st.spinner("正在处理视频，请稍候..."):
            # 进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 实例化模型
            try:
                input_size = 258
                hidden_size = 64
                num_classes = len(gestures)
                
                # 创建模型实例
                model = CustomLSTM(input_size, hidden_size, num_classes)
                
                model = torch.load(model_path, weight_only=False)
                
                model.eval()
                
            except Exception as e:
                st.error(f"加载模型时出错: {str(e)}")
                st.info("继续使用默认模型进行演示")
                # 创建默认模型
                model = CustomLSTM(258, 64, len(gestures))
                model.eval()
            
            # 处理视频
            cap = cv2.VideoCapture(video_path)
            
            # 准备输出视频（如果需要）
            if save_output:
                output_path = f"processed_{file_name}"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 初始化变量
            sequence = []
            predictions_history = []  # 存储所有预测结果
            processed_frames = []
            frame_count = 0
            
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
                    status_text.text(f"正在处理第 {frame_count}/{total_frames} 帧")
                    
                    # 检测关键点
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # 绘制关键点
                    if show_video:
                        draw_styled_landmarks(image, results)
                    
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
                                'frame': frame_count,
                                'class': pred_class,
                                'confidence': confidence,
                                'timestamp': frame_count / fps
                            })
                            
                            # 在图像上显示预测结果
                            if show_video:
                                cv2.putText(image, f"预测: {pred_class}", 
                                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           1, (0, 255, 0), 2, cv2.LINE_AA)
                                cv2.putText(image, f"置信度: {confidence:.1f}%", 
                                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                           1, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        except Exception as e:
                            st.warning(f"第 {frame_count} 帧预测出错: {str(e)}")
                    
                    # 保存处理后的帧
                    if show_video:
                        processed_frames.append(image)
                    
                    # 写入输出视频
                    if save_output:
                        out.write(image)
            
            # 释放资源
            cap.release()
            if save_output:
                out.release()
            
            # 更新完成状态
            progress_bar.progress(1.0)
            status_text.text("✅ 处理完成！")
            
            with col2:
                st.subheader("📊 分析结果")
                
                # 打印文件名称
                st.info(f"**文件名称:** {file_name}")
                
                # 检查是否有预测结果
                if predictions_history:
                    # 获取最后30帧的预测结果
                    last_30_predictions = predictions_history[-30:] if len(predictions_history) >= 30 else predictions_history
                    
                    st.success("✅ 找到手部动作！")
                    st.write(f"**总预测帧数:** {len(predictions_history)}")
                    
                    # 分析最后30帧的预测
                    if last_30_predictions:
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
                            
                            st.subheader("🎯 最终预测分类")
                            st.success(f"**预测结果:** {final_prediction}")
                            st.write(f"**置信度:** {confidence_score:.1f}%")
                            st.write(f"**在最后{len(last_30_predictions)}帧中出现次数:** {most_common_class[1]}次")
                            
                            # 显示详细统计
                            st.subheader("📈 详细统计")
                            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / len(last_30_predictions)) * 100
                                st.write(f"- **{cls}**: {count}次 ({percentage:.1f}%)")
                        else:
                            st.warning("未能确定最终分类")
                    else:
                        st.warning("没有足够的预测数据进行分类")
                    
                    # 显示预测历史表格
                    if len(predictions_history) > 0:
                        st.subheader("📋 预测历史")
                        
                        # 创建简化的历史视图
                        history_data = []
                        for i, pred in enumerate(predictions_history[-20:]):  # 只显示最后20条
                            history_data.append({
                                '帧号': pred['frame'],
                                '时间戳': f"{pred['timestamp']:.1f}s",
                                '预测分类': pred['class'],
                                '置信度': f"{pred['confidence']:.1f}%"
                            })
                        
                        st.dataframe(history_data)
                
                else:
                    st.warning("⚠️ 未检测到手部动作，请确保视频中包含清晰的手部动作")
                
                # 显示处理后的视频帧
                if show_video and processed_frames:
                    st.subheader("🎥 处理后的视频预览")
                    
                    # 选择显示一些关键帧
                    display_frames = []
                    if len(processed_frames) > 10:
                        step = len(processed_frames) // 9
                        for i in range(0, len(processed_frames), step):
                            if len(display_frames) < 9 and i < len(processed_frames):
                                display_frames.append(processed_frames[i])
                    else:
                        display_frames = processed_frames
                    
                    # 显示帧网格
                    cols = st.columns(3)
                    for idx, frame in enumerate(display_frames[:9]):
                        with cols[idx % 3]:
                            # 调整图像大小以适应显示
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame_rgb)
                            img.thumbnail((200, 200))
                            st.image(img, caption=f"帧 {idx*step if len(processed_frames)>10 else idx+1}")
                
                # 下载处理后的视频
                if save_output and os.path.exists(output_path):
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 下载处理后的视频",
                            data=file,
                            file_name=output_path,
                            mime="video/mp4"
                        )
            
            # 清理临时文件
            try:
                os.unlink(video_path)
                if save_output and os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass

# 添加说明
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    ### 使用方法：
    1. **上传视频**：点击"Browse files"上传MP4格式的视频文件
    2. **配置参数**：在侧边栏设置手势类别和模型路径
    3. **开始处理**：点击"开始处理视频"按钮进行分析
    4. **查看结果**：系统将显示预测结果和处理后的视频预览
    
    ### 注意事项：
    - 确保视频中包含清晰的手部动作
    - 手势类别需要与训练模型时的类别一致
    - 系统会分析视频最后30帧的预测结果来确定最终分类
    - 处理时间取决于视频长度和计算机性能
    
    ### 技术说明：
    - 使用MediaPipe进行人体关键点检测
    - 使用LSTM神经网络进行时序动作识别
    - 分析最后30帧的预测结果来确定最终手势分类
    """)

# 添加页脚
st.markdown("---")
st.markdown("👋 **手势识别应用** | 基于MediaPipe和LSTM的动作识别系统")
