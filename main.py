from ultralytics import YOLO

if __name__ == '__main__':
    # load model
    model = YOLO('yolov8m.pt')
    # train
    results = model.train(data='my_dataset_yolo/data.yaml', epochs=100, imgsz=640, model='yolov8m.pt')
