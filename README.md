# Yolo V5, V8 and NAS

<div>!pip install super-gradients #for Yolo-NAS</div>
<div>!pip install ultralytics</div>

## Yolo V5

<div>➢ from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")</div>
<div>➢ os.environ["DATASET_DIRECTORY"] = "../apple_datasets"</div>
<div>➢rf = Roboflow(api_key="your_key")
project = rf.workspace("avanti-nlyef").project("flowers-jpdha")
dataset = project.version(2).download("yolov5")</div>
<div>➢ !python detect.py --weights /home/nik/Downloads/flowers.pt --img 416 --conf 0.1 --source flower.mp4
#Results saved to runs/detect/exp9</div>

## Yolo V8

<div>➢ from ultralytics import YOLO</div>
<div>➢ os.environ["DATASET_DIRECTORY"] = "flowers"</div>
<div>➢ from roboflow import Roboflow</div>
<div>rf = Roboflow(api_key="N3RIafLUvo5XjHHr9CkK")</div>
<div>project = rf.workspace("avanti-nlyef").project("flowers-jpdha")</div>
<div>dataset = project.version(2).download("yolov8")</div>
<div>➢ !yolo task=detect mode=val model=yolov8s.pt data={dataset.location}/data.yaml batch=10 imgsz=640</div>
<div>Results saved to runs\detect\val5</div>
<div>
  
</div>
<div>#to predict objects using webcam</div>
<div>#model2 = YOLO('yolov8n.pt')</div>
<div>#result2=model2.predict(source='0',show=True)</div>
<div>#print(result2)</div>

## Yolo-NAS

<div>➢ from super_gradients.training import models</div>
<div>import super_gradients</div>
<div>from super_gradients.training import models</div>
<div>from super_gradients.common.object_names import Models</div>
<div>➢ device = torch.device("cpu")</div>
<div>yolo_nas_s = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")</div>
<div>out = yolo_nas_s.predict("office.jpg")</div>
<div>out.save("office_result-yolo_nas_s.jpg")</div>
<div>yolo_nas_s.predict("office.jpg").show()</div>

![off](https://github.com/AvantiBuche/Yolo_V5_V8_NAS/blob/0d3ec993a6c8855fe8d564d07e540c16a48eb0f9/off.png)


<div>➢ yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco")</div>
<div>yolo_nas.predict("office.jpg").show()</div>
<div>➢ yolo_nas.predict("market.jpg").show()</div>

![mar](https://github.com/AvantiBuche/Yolo_V5_V8_NAS/blob/0d3ec993a6c8855fe8d564d07e540c16a48eb0f9/mar.png)


<div>➢ yolo_nas.predict("traffic.jpg").show()</div>

![tra](https://github.com/AvantiBuche/Yolo_V5_V8_NAS/blob/0d3ec993a6c8855fe8d564d07e540c16a48eb0f9/tra.png)

