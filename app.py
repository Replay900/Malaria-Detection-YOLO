from flask import Flask, render_template, request, send_file
import os
from ultralytics import YOLO
import cv2
import uuid

app = Flask(__name__)

# Cargar el modelo YOLOv11
model = YOLO("model.pt")

# Clases
CLASSES = ["trophozoite", "ring", "schizont", "gametocyte"]

# Rutas
UPLOAD_FOLDER = "images"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    filename = str(uuid.uuid4()) + "_" + imagefile.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    result_path = os.path.join(RESULT_FOLDER, filename)

    # Guardar imagen
    imagefile.save(image_path)

    # Realizar predicción
    results = model.predict(image_path, fuse=False)[0]

    # Leer imagen
    image = cv2.imread(image_path)

    # Dibujar cajas
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{CLASSES[cls]}: {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Guardar resultado
    cv2.imwrite(result_path, image)

    return render_template("index.html", result_img=result_path)
