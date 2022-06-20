import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    # Opencv는 기본적으로 BGR의 형태로 변환하여 데이터를 가져오기 때문.
    image.flags.writeable = False  # Image is no longer writeable
    # Make an array immutable(read-only) -> 성능 향상을 위해서.

    # 탐지
    results = model.process(image)  # Make predictionF
    # 미디어 파이프를 사용하여 감지.

    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results): # landmark(x,y,z) 추출. # 이거 없어도 되는거 아닌가? 뒤에 업데이트 있으니.
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw right hand con


def draw_styled_landmarks(image, results):
    # draw_landamarks Update.
    # Draw face connections
    # 1st : color_landmark
    # 2nd : color to connection
    # 색 그리고 라인 두께, 점 반지름은 몇 번 돌려보고 보기 좋은 걸로 설정.
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results): # 평탄화 작업을 통해 하나의 array로 랜드마크들을 통합시킴.
    # len(pose) = 33(flatten 전), len(pose) = 132(33*4) (flatten 후)
    # pose.shape = (33,4) -> 33가지의 landmark를 각각 4가지 씩(flatten 전), pose.shape = (132,0) (flatten 후)
    # 만약에 pose를 인식했을 경우 result.pose_landmarks 아니면 empty numpy array 출력(사이즈 맞추어서) (아니면 에러뜸)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    # 모든 요소들을 하나로 모아서 수어 인식에 사용.
    return np.concatenate([pose, face, lh, rh])


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['안녕'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # min_detection_confidence=0.5 -> 탐지 신뢰도(0.5 = 기본값)
    # min_tracking_confidence=0.5 -> 추적 신회도(0.5 = 기본값)
    # -> 추적 신뢰값을 높게 설정하면 솔루션의 견고함이 증가하지만 대기시간도 증가. -> 얼마나 증가하는지 체크 해도 ㄱㅊ을듯.
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences): # 각각 반복(30)
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                #print(results)

                

                # Draw landmarks
                draw_styled_landmarks(image, results)

                font=ImageFont.truetype("fonts/gulim.ttc",30)
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)

                # NEW Apply wait logic # delay를 주기 위해서.
                if frame_num == 0:
                    draw.text((120,200), 'STARTING COLLECTION', font=font, fill=(0, 255,0))
                    # cv 시작되는 지점. x = 120, y = 200
                    font=ImageFont.truetype("fonts/gulim.ttc",15)
                    draw.text((15,12), 'Colloecting frames for {} Video Number {}'.format(action, sequence) , font=font, fill=(0, 0 ,255))
                    # 비디오 넘버 지정.

                    # Show to screen
                    image= np.array(image)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    font=ImageFont.truetype("fonts/gulim.ttc",15)
                    draw.text((15,12), 'Colloecting frames for {} Video Number {}'.format(action, sequence) , font=font, fill=(0, 0 ,255))            
                    # Show to screen
                    image= np.array(image)
                    cv2.imshow('OpenCV Feed', image)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'): # q로 탈출.
                    break

    cap.release()
    cv2.destroyAllWindows()
