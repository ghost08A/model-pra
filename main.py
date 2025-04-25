from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from PIL import Image
import io

app = FastAPI()

models = {}
model_names = {
    1: 'ConvNeXtTiny',
    2: 'DenseNet121',
    3: 'ResNet50',
    4: 'MobileNetV2'
}

output_names = ['label_pra', 'label_wat', 'label_pim', 'label_year', 'name']
encoders = {}
threshold = 0.7


@app.on_event("startup")
def load_resources():
    global models, encoders
    # โหลดโมเดลทั้งหมดในตอนที่แอปเริ่มต้น
    models[1] = load_model('model_pra/ConvNeXtTiny.keras')
    models[2] = load_model('model_pra/DenseNet121.keras')
    models[3] = load_model('model_pra/ResNet50.keras')
    models[4] = load_model('model_pra/MobileNetV2.keras')
    
    # โหลด encoder ที่ใช้ในการแปลง label
    encoders.update(joblib.load('model_pra/encoders.pkl'))



# ฟังก์ชันประมวลผลภาพ
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.post("/predict/")
async def predict(file: UploadFile = File(...), model_id: int = Form(...)):
    try:
        # ตรวจสอบไฟล์เป็นภาพมั้ย
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are allowed.")
        
        # ตรวจสอบขนาดไฟล์ไม่เกิน 10MB
        img_bytes = await file.read()
        if len(img_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image size exceeds 10MB limit.")
        
        # ตรวจสอบว่า model_id มีมั้ย
        if model_id not in models:
            return JSONResponse(status_code=400, content={"message": "Invalid model ID"})
        
        # แปลงภาพให้พร้อมให้โมเดลทำนาย
        img_tensor = preprocess(img_bytes)

        # เลือกโมเดลที่ผู้ใช้เลือกมา
        model = models[model_id]
        preds = model.predict(img_tensor)

        predicted_labels = {}
        total_conf = 0.0

        # ทำนาย
        for i, pred in enumerate(preds):
            pred = pred[0]
            max_conf = np.max(pred)
            total_conf += max_conf

            class_index = np.argmax(pred)
            label = encoders[output_names[i]].inverse_transform([class_index])[0]
            predicted_labels[output_names[i]] = label

        # หาความมั่นใจ
        average_conf = total_conf / len(output_names)

        # เจอ 'not_pra' คือไม่รู้จัก
        if average_conf < threshold or "not_pra" in predicted_labels.values():
            return JSONResponse(status_code=200, content={"pra": "ไม่อยู่ในฐานข้อมูล"})
        
        # ผลลัพธ์ที่ได้จากการทำนาย
        response = {
            "used_model": model_names[model_id],
            "confidence": f"{average_conf * 100:.2f}%"
        }

        # เพิ่ม labels ที่ทำนายได้เข้าไปในผลลัพธ์
        for key in predicted_labels:
            response[key] = predicted_labels[key]

        return JSONResponse(content=response)
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
