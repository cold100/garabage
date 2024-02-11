import cv2
from ultralytics import YOLO
import random


def draw_bounding_boxes(image, results):
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    for box, clss in zip(boxes, classes):
        # Generate a random color for each object based on its ID
        random.seed(int(clss) + 8)
        color = (random.randint(0, 0), random.randint(255, 255), random.randint(0, 255))

        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3],), color, 2)
        cv2.putText(
            image,
            f"{model.model.names[clss]}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (50, 255, 50),
            2,
        )
    return image


def process_image_with_tracking(model, image_path, window_width, window_height):
    image = cv2.imread(image_path)

    results = model.predict(image)

    draw_bounding_boxes(image, results)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)  # Создаем окно с возможностью изменения размера
    cv2.resizeWindow("image", window_width, window_height)  # Устанавливаем фиксированный размер окна

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


# Example usage:
model = YOLO('runs/detect/train/weights/best.pt')
model_detect = YOLO('runs/detect/train/weights/best.pt')
model.fuse()
model_detect.fuse()
results = process_image_with_tracking(model, "test.jpg", 800, 600)

