from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf 
from utils.network import get_model
from utils.mediapipetools import dl_history_plot
import os 
import numpy as np 
import random

# Random Seed 고정
def set_seed(seed=42):
    random.seed(seed)                  # Python 랜덤 시드
    np.random.seed(seed)               # Numpy 시드
    tf.random.set_seed(seed)
    # torch.manual_seed(seed)            # PyTorch 시드
    # torch.cuda.manual_seed(seed)       # CUDA 시드
    # torch.cuda.manual_seed_all(seed)   # 다중 GPU 시드
    # torch.backends.cudnn.deterministic = True  # CuDNN 결과 일관성
    # torch.backends.cudnn.benchmark = False     # CuDNN 최적화 비활성화 (일관성 유지)

set_seed(123)



##################################### 데이터 로드 #####################################

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['nothing', 'ready', 'stop'])

# Thirty videos worth of data
no_sequences = 30
# Videos are going to be 30 frames in length
sequence_length = 15

label_map = {label:num for num, label in enumerate(actions)}


sequences, labels = [], []
for action in actions:
    if action == 'nothing':
        no_sequences = 90
    else:
        no_sequences = 30
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        
X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


################################# 모델 로드 ####################################

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

model = get_model(actions, sequence_length, X.shape[-1])

# 모델 컴파일
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])


################################### 모델 학습 ####################################

# 모델 학습
history = model.fit(X_train, y_train, 
          epochs=1000, 
          batch_size=512,
          validation_split=0.2,
          callbacks=[tb_callback]).history

res = model.predict(X_test)

print(res)

model.save('action.h5')

dl_history_plot(history)