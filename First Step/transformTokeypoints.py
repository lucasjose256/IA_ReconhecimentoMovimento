import cv2
import pandas as pd
import os
from pathlib import Path
import mediapipe as mp

# Caminho para o diretório das imagens e o arquivo CSV
image_dir = "/Users/lucasbarszcz/Downloads/Plank_detection.v1i.tensorflow/train/"  # Substitua pelo caminho do diretório
csv_file = "keypoints_output.csv"
output_dir = "/Users/lucasbarszcz/Downloads/Plank_detection.v1i.tensorflow/keypoints"  # Diretório para salvar as imagens com keypoints desenhados

# Criar diretório de saída, se não existir
os.makedirs(output_dir, exist_ok=True)

# Carregar o arquivo CSV
df = pd.read_csv(csv_file)

# Configurações do MediaPipe para as conexões do esqueleto
mp_pose = mp.solutions.pose
pose_connections = mp_pose.POSE_CONNECTIONS


# Função para desenhar keypoints e conexões
def draw_keypoints(image, landmarks, draw_connections=True):
    height, width = image.shape[:2]

    # Desenhar keypoints como círculos
    for idx in range(33):  # MediaPipe Pose tem 33 landmarks
        x = landmarks[f"landmark_{idx}_x"] * width
        y = landmarks[f"landmark_{idx}_y"] * height
        visibility = landmarks[f"landmark_{idx}_visibility"]

        # Desenhar apenas se a visibilidade for alta o suficiente
        if visibility > 0.3:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Desenhar conexões do esqueleto, se solicitado
    if draw_connections:
        for connection in pose_connections:
            start_idx, end_idx = connection
            x1 = landmarks[f"landmark_{start_idx}_x"] * width
            y1 = landmarks[f"landmark_{start_idx}_y"] * height
            x2 = landmarks[f"landmark_{end_idx}_x"] * width
            y2 = landmarks[f"landmark_{end_idx}_y"] * height
            vis1 = landmarks[f"landmark_{start_idx}_visibility"]
            vis2 = landmarks[f"landmark_{end_idx}_visibility"]

            # Desenhar linha apenas se ambos os pontos tiverem visibilidade suficiente
            if vis1 > 0.5 and vis2 > 0.5:
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return image


# Iterar sobre as linhas do DataFrame
for _, row in df.iterrows():
    image_name = row["image_name"]
    image_path = Path(image_dir) / image_name

    # Carregar a imagem
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        continue

    # Desenhar keypoints e conexões
    image_with_keypoints = draw_keypoints(image, row)

    # Salvar a imagem com os keypoints desenhados
    output_path = Path(output_dir) / f"keypoints_{image_name}"
    cv2.imwrite(str(output_path), image_with_keypoints)
    print(f"Imagem processada salva em: {output_path}")

print("Processamento concluído.")