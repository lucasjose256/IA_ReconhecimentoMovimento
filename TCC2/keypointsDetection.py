
import cv2
import pandas as pd
import os
from pathlib import Path
import mediapipe as mp

class KeypointDetector:
    def __init__(self, image_dir, csv_file, output_dir):
        """
        Inicializa o KeypointDetector com os caminhos necessários.

        Args:
            image_dir (str): Caminho para o diretório das imagens.
            csv_file (str): Caminho para o arquivo CSV com os keypoints.
            output_dir (str): Caminho para o diretório de saída das imagens processadas.
        """
        self.image_dir = Path(image_dir)
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = pd.read_csv(self.csv_file)
        self.mp_pose = mp.solutions.pose
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS

    def draw_keypoints(self, image, landmarks, draw_connections=True):
        """
        Desenha keypoints e conexões em uma imagem.

        Args:
            image (numpy.ndarray): Imagem onde os keypoints serão desenhados.
            landmarks (pandas.Series): Dados dos keypoints extraídos do CSV.
            draw_connections (bool): Se True, desenha as conexões entre os keypoints.

        Returns:
            numpy.ndarray: Imagem com os keypoints desenhados.
        """
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
            for connection in self.pose_connections:
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

    def process_images(self):
        """
        Processa todas as imagens do diretório, desenhando keypoints e salvando os resultados.
        """
        for _, row in self.df.iterrows():
            image_name = row["image_name"]
            image_path = self.image_dir / image_name

            # Carregar a imagem
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Erro ao carregar a imagem: {image_path}")
                continue

            # Desenhar keypoints e conexões
            image_with_keypoints = self.draw_keypoints(image, row)

            # Salvar a imagem com os keypoints desenhados
            output_path = self.output_dir / f"keypoints_{image_name}"
            cv2.imwrite(str(output_path), image_with_keypoints)
            print(f"Imagem processada salva em: {output_path}")

        print("Processamento concluído.")

# Exemplo de uso
if __name__ == "__main__":
    image_dir = "/Users/lucasbarszcz/Downloads/Plank_detection.v1i.tensorflow/train/"
    csv_file = "keypoints_output.csv"
    output_dir = "/Users/lucasbarszcz/Desktop/TCC/keypoints"

    detector = KeypointDetector(image_dir,  output_dir,csv_file)
    detector.process_images()
