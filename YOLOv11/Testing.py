from ultralytics import YOLO

# model = YOLO('/home/gen79net/Downloads/Image Detection/YOLOv11/runs/segment/train2/weights/best.pt')
model = YOLO('/home/gen79net/Downloads/Image Detection/YOLOv11/YOLO_model/best.pt')
if __name__ == "__main__":
    result = model(source = '/home/gen79net/Downloads/Image Detection/Segmented Images/train/images/1e361b14-image85.jpg',
                       show_boxes = False)

    for r in result:
        mask_array = r.masks
        print(type(mask_array))



