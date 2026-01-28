import cv2
import base64
import numpy as np
import mediapipe as mp
import os
from fastapi import FastAPI, WebSocket
import uvicorn
from keras.models import load_model

app = FastAPI()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

model_mlp = None
model_cnn = None
classes = ["A", "B", "C"]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

if os.path.exists("classes.npy"):
    classes = np.load("classes.npy", allow_pickle=True)

try:
    model_mlp = load_model("modelo_mlp.keras")
    print("MLP carregada")
except:
    print("Erro ao carregar MLP")

try:
    model_cnn = load_model("modelo_cnn1d.keras")
    print("CNN carregada")
except:
    print("Erro ao carregar CNN")