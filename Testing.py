from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO(r"C:\Users\eee_admin\ProjectDiss\Trial5\weights\best.pt")

#Loading the image
results = model("C:/Users/eee_admin/Downloads/snow.jpg")

# Get the annotated image array
annotated_image = results[0].plot()  # YOLOv8 returns a list of results; we access the first one

#Resize for consistent display
annotated_image = cv2.resize(annotated_image, (640,640))

# Display the image using OpenCV
cv2.imshow("Detected Objects", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()