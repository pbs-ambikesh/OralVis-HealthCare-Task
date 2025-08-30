# train.py

from ultralytics import YOLO

# --- 1. Load a Pretrained Model ---
model = YOLO('yolov8s.pt')

# --- 2. Train the Model ---
# The model.train() function starts the training process.
# Passing our data.yaml file to it and specify other training parameters.
results = model.train(
   data='data.yaml',        
   imgsz=640,               # Image size for training (640x640 pixels)
   epochs=50,               # Number of times to train on the entire dataset
   batch=8,                 
   name='oralvis_yolov8s_run' 
)

print("Training finished!")