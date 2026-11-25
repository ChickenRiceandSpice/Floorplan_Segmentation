from ultralytics import YOLO

model = YOLO('/home/gen79net/Downloads/Image Detection/Segmented Images/yolo11m-seg.pt')
model.train(data = '/home/gen79net/Downloads/Image Detection/Segmented Images/dataset_custom.yaml', imgsz = 2000,epochs = 30, device = 0)







