import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# Configuração do modelo
def setup_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

# Inicializa o modelo
predictor = setup_model()

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)  # Use 0 para webcam, ou o caminho do vídeo

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte o frame para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Faz a previsão
    outputs = predictor(frame_rgb)

    # Visualiza os resultados
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE)
    vis_frame = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Exibe o frame com os keypoints detectados
    cv2.imshow("Keypoint Detection", vis_frame.get_image()[:, :, ::-1])

    # Para sair, pressione 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
