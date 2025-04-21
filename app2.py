import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# โหลดโมเดลที่ฝึกไว้
MODEL_PATH = "models/lstm_model.h5"  # เปลี่ยนเป็น path ของโมเดลที่คุณฝึกไว้
model = load_model(MODEL_PATH)

# รายชื่อคำที่โมเดลรองรับ
LABELS = ["Hello", "Yes", "No", "Thank you", "Please"]  # แก้ไขตามที่คุณใช้

# ฟังก์ชันประมวลผลภาพ
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # ปรับขนาดให้ตรงกับที่ใช้ตอนฝึก
    img = img / 255.0  # ปรับให้อยู่ในช่วง 0-1
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# ฟังก์ชันสำหรับเลือกไฟล์และทำนาย
def predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    predicted_label = LABELS[np.argmax(prediction)]
    
    result_label.config(text=f"Prediction: {predicted_label}")

# UI ด้วย Tkinter
root = tk.Tk()
root.title("Sign Language Translator")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(pady=10)

btn_select = tk.Button(frame, text="Select Image", command=predict)
btn_select.pack()

result_label = tk.Label(frame, text="Prediction: ", font=("Arial", 14))
result_label.pack()

root.mainloop()
