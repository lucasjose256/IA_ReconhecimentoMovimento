import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import detectron2.data.transforms as T
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)


img_path = 'Users/lucasbarszcz/PycharmProjects/ReconhecimentoMovimento/First Step/plank_images_103.png'
image = cv2.imread(img_path)
outputs = predictor(image)

from detectron2.utils.visualizer import Visualizer
v = Visualizer(image[:, :, ::-1], scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Skeleton Keypoints", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
