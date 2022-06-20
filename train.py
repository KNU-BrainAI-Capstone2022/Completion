import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
# 사이킷런 -> 머신러닝 라이브러리.(기계학습)
# 자신이 하고 싶은 분석(분류/회귀/클러스터링 등)에 대해서 적당한 모델을 선택할때 도움.
# train_test_split -> model selection 안에 있는 데이터 분할을 위한 함수.


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['ㄱ','감사합니다', 'ㄴ', 'ㄷ','안녕','친구'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Preprocess Data and Create Labels and Features
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
# sequences : feature data or x data
# labels : label data or y data
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y= to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05)
# test size : 5%

# tensorboard을 plt 대신 사용함./ 이거 위치는 수정 좀 해야할 듯? 돌아갈라나
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential() # 순차적 모델 구축

# LSTM 3 Layers
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
# return_sequences = True -> 다음 레이어가 반환 값을 필요로 하기 때문.
# input_shape -> 30 frame, 468*3 + 33*4 + 21*3 + 21*3 = 1662
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Dense 3 layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
# actions.shape[0] = 3

#res = [.7, 0.2, 0.1]
#actions[np.argmax(res)]
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=250, callbacks=[tb_callback])

model.save('action.h5')
del model
