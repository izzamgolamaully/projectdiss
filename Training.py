import multiprocessing

def main():
    from ultralytics import YOLO

    # Load the YOLOv8 model
    model = YOLO(r"C:\Users\eee_admin\ProjectDiss\Trial5\weights\last.pt")

    # Run inference on an image
    # device=0 should be default if not specified when YOLO set up for GPU. Use device='cpu' to force CPU
    model.train(
        data="C:/Project/Solar Panel Test/data.yaml",
        project="ProjectDiss",
        name = "Trial",
        epochs = 100,
        batch= 16,
        imgsz = 640,
        device =0, 
        optimizer = "AdamW",
        lr0 = 0.01,
        #patience = 10,
        augment = True,
        pretrained = True,
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()