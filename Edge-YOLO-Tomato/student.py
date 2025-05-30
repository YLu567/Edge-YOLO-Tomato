from ultralytics import YOLO

# 加载自定义模型
model = YOLO(
    '/home/featurize/work/ultralytics-3.31/s_CAPA+c.yaml').load('/home/featurize/work/ultralytics-3.31/runs/detect/Edge-YOLO-Tomato2/weights/best.pt')  # 替换为你的YAML路径

# 训练模型
results = model.train(
    data='/home/featurize/work/ultralytics-3.31/data.yaml',  # 数据集配置文件
    epochs=50,
    imgsz=640,
    batch=16,
    name='Edge-YOLO-Tomato'
)