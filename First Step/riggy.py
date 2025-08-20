import mmpose.apis
import cv2


config_file = 'configs/hrnet/hrnet_w32_coco_256x192.py'
checkpoint_file = 'checkpoints/hrnet_w32_coco_256x192.pth'
model = init_pose_model(config_file, checkpoint_file)

img_path = '../imagem1.jpeg'
image = cv2.imread(img_path)

person_bboxes = [{'bbox': [50, 50, 200, 400]}]

pose_results, _ = inference_top_down_pose_model(
    model, image, person_bboxes, format='xywh'
)

vis_image = vis_pose_result(model, img_path, pose_results, dataset='TopDownCocoDataset')
cv2.imshow('Result', vis_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
