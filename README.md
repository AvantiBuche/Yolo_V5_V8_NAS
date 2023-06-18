# Yolo V5, V8 and NAS

<div>!pip install super-gradients #for Yolo-NAS</div>
<div>!pip install ultralytics</div>

## Yolo V5

➢ from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")
➢ os.environ["DATASET_DIRECTORY"] = "../apple_datasets"
➢rf = Roboflow(api_key="your_key")
project = rf.workspace("avanti-nlyef").project("flowers-jpdha")
dataset = project.version(2).download("yolov5")
➢ !python detect.py --weights /home/nik/Downloads/flowers.pt --img 416 --conf 0.1 --source flower.mp4
#Results saved to runs/detect/exp9

## Yolo V8

➢ from ultralytics import YOLO
➢ os.environ["DATASET_DIRECTORY"] = "flowers"
➢ from roboflow import Roboflow
rf = Roboflow(api_key="N3RIafLUvo5XjHHr9CkK")
project = rf.workspace("avanti-nlyef").project("flowers-jpdha")
dataset = project.version(2).download("yolov8")
➢ !yolo task=detect mode=val model=yolov8s.pt data={dataset.location}/data.yaml batch=10 imgsz=640
Results saved to runs\detect\val5

#to predict objects using webcam
#model2 = YOLO('yolov8n.pt')
#result2=model2.predict(source='0',show=True)
#print(result2)

## Yolo-NAS

➢ from super_gradients.training import models
import super_gradients
from super_gradients.training import models
from super_gradients.common.object_names import Models
➢ device = torch.device("cpu")
yolo_nas_s = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")
out = yolo_nas_s.predict("office.jpg")
out.save("office_result-yolo_nas_s.jpg")
yolo_nas_s.predict("office.jpg").show()


➢ 
➢ 
