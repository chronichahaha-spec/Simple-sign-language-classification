import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import torch
from torch import nn
import mediapipe as mp
from PIL import Image

# page setting
st.set_page_config(
    page_title="Sign Language Video Classfication",
    page_icon="👋",
    layout="wide"
)

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    # process image and detect key gesture points
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

# LSTM model framework
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

# gestures prediction classes
gestures = np.array(['polis','nasi','abang','apa','hari','ribut','pukul','beli','emak','perlahan'])

# Load the trained model
model_path = "model.pth"
input_size = 258
hidden_size = 64
num_classes = len(gestures)


model = torch.load(model_path,weights_only=False, map_location=torch.device('cpu'))
model.eval()

# main title
st.title("👋 Sign Language Video Classification")

# first message: reminder to upload videos
st.markdown("### Please upload your sign language to get AI model predictiom")

# button to upload videos
uploaded_file = st.file_uploader("Select the video file", type=['mp4'])

if uploaded_file is not None:
    # save as temporary file to pass into LSTM model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    # read video information
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # second message: show video information
    st.markdown("### Video Information")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**File Name:** {uploaded_file.name}")
    with col2:
        st.info(f"**Duration:** {duration:.2f}seconds")
    
    # Button to start model prediction
    if st.button("Start Prediction", type="primary"):
        with st.spinner("Processing the Video..."):
            cap = cv2.VideoCapture(video_path)
            sequence = []
            predictions_history = []
            frame_count = 0
            
            # progress bar
            progress_bar = st.progress(0)
            
            # media pipe model
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as holistic:
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # update progess bar
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(min(progress, 1.0))
                    
                    # key point detection
                    _, results = mediapipe_detection(frame, holistic)
                    
                    # draw landmarks
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]  # keep a sequence queue with the last updated 30 frame
                    
                    if (results.left_hand_landmarks or results.right_hand_landmarks) and len(sequence) == 30:
                        try:
                            input_data = torch.tensor(
                                np.expand_dims(sequence, axis=0), 
                                dtype=torch.float32
                            )
                            
                            # model prediction
                            with torch.no_grad():
                                res = model(input_data)
                            
                            probabilities = torch.softmax(res, dim=1)
                            max_prob, max_idx = torch.max(probabilities, dim=1)
                            
                            pred_class = gestures[max_idx.item()]
                            confidence = max_prob.item() * 100
                            
                            # prediction results list strated by the first 30 frame
                            predictions_history.append({
                                'class': pred_class,
                                'confidence': confidence
                            })
                            
                        except Exception as e:
                            pass
            
            # relase cap reader
            cap.release()
            progress_bar.progress(1.0)
            
            # third message: show prediction results
            st.markdown("### prediction results")
            
            first_prediction = predictions_history[0]       
            st.success(f"**prediction results:** {first_prediction['class']}")
            st.info(f"**confidence:** {first_prediction['confidence']:.1f}%")
            
            # clean the temporary file
            try:
                os.unlink(video_path)
            except:
                pass

# readme
st.markdown("---")
st.markdown("*Please upload your sign language vido, AI model will classify the gestures*")
