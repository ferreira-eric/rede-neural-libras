import cv2
import base64
import numpy as np
from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

# Aqui você carregaria suas Redes Neurais
# rede1 = HandDetector()
# rede2 = LibrasClassifier()

@app.websocket("/ws")
async def libras_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            print("Nova conexão WebSocket")
            # 1. Recebe o dado do React
            data = await websocket.receive_json()
            image_b64 = data.get("image")
            
            if image_b64:
                # 2. Converte Base64 para imagem OpenCV
                format, imgstr = image_b64.split(';base64,') 
                nparr = np.frombuffer(base64.b64decode(imgstr), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # 3. Processamento (Suas duas redes entram aqui)
                # mao_recortada = rede1.detect(frame)
                # resultado = rede2.classify(mao_recortada)
                resultado = "A" # Simulação de predição

                # 4. Envia a resposta de volta
                await websocket.send_json({"letter": resultado})
    except Exception as e:
        print(f"Conexão encerrada: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)