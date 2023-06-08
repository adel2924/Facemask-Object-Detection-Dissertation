# Training

from ultralytics import YOLO

model = YOLO('yolov8m.pt')

if __name__ == '__main__':
    # Train
    model.train(
        data='masks.yaml',
        batch=-1,
        epochs=10,
        device='0',
    )