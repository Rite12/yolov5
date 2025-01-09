import sys
import os
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# Konfigurasi YOLOv5
sys.path.append("F:/Codes/yolov5")
device = select_device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend("F:/Codes/yolov5/yolov5l.pt", device=device, fp16=True)
stride, names, pt = model.stride, model.names, model.pt

# URL Stream ATCS Alun-Alun Kota Bandung
stream_url = "https://pelindung.bandung.go.id:3443/video/HIKSVISION/asiaafrika.m3u8"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open the video stream.")
    sys.exit()

# Inisialisasi Counter Kendaraan
vehicle_count = {'car': 0, 'truck': 0, 'motorcycle': 0}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    # Resize dan preprocess frame
    img = letterbox(frame, stride=stride, auto=True)[0]
    img = torch.from_numpy(img).to(device).float()
    img = img.permute(2, 0, 1).unsqueeze(0)  # HWC ke CHW
    img /= 255.0  # Normalisasi

    # Inference YOLOv5
    pred = model(img)
    pred = non_max_suppression(pred, 0.4, 0.5, classes=[2, 3, 7])  # Filter untuk mobil, truk, motor

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = names[int(cls)]
                if label in vehicle_count:
                    vehicle_count[label] += 1  # Hitung kendaraan

    # Tampilkan frame
    for k, v in vehicle_count.items():
        cv2.putText(frame, f"{k}: {v}", (10, 30 + list(vehicle_count.keys()).index(k) * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Klasifikasi Kendaraan", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Simpan Data Kendaraan
with open('vehicle_data.csv', 'w') as f:
    f.write("Jenis,Kuantitas\n")
    for k, v in vehicle_count.items():
        f.write(f"{k},{v}\n")