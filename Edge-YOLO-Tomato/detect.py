import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
 
 
if __name__ == '__main__':
    model = YOLO(' /home/featurize/work/ultralytics-3.31/runs/detect/s_0.5-CAPA/weights/best.pt') # select your model.pt path
    model.val(source='/home/featurize/work/ultralytics-3.31/predictdata',
                  imgsz=640,
                  project='runs/detect/predict',
                  name='Edge-YOLO-Tomato',
                  save=True,
                  conf=0.6,
                  # iou=0.7,
                  # agnostic_nms=True,
                  #visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )