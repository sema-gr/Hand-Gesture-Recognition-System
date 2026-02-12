from insightface.app import FaceAnalysis
import numpy as np
import cv2

class FaceEmbedder:
    """
    Клас для отримання 512-вимірних face embeddings через InsightFace (ArcFace).
    """

    def __init__(self, device: str = "cpu"):
        """
        device: "cpu" або "cuda"
        """
        self.device = device
        self.app = FaceAnalysis(
            name="buffalo_l",  # модель високої якості
            providers=self._get_providers()
        )
        self.app.prepare(ctx_id=self._get_ctx_id())

    def _get_ctx_id(self) -> int:
        # -1 означає CPU, 0 - GPU
        return 0 if self.device == "cuda" else -1

    def _get_providers(self):
        # Порядок провайдерів для onnxruntime
        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def get_embeddings(self, frame_bgr: np.ndarray):
        """
        Отримує список знайдених облич із bounding box і embedding.

        Аргументи:
            frame_bgr: кадр у форматі BGR (OpenCV)

        Повертає:
            Список словників:
            [
              {
                "bbox": [x1, y1, x2, y2],
                "embedding": np.ndarray з формою (512,)
              },
              ...
            ]
        """
        faces = self.app.get(frame_bgr)

        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": face.embedding
            })

        return results

    def draw_faces(self, frame_bgr, color=(0, 255, 0)):
        """
        Промальовує рамки на кадрі (корисно для дебагу)
        """
        faces = self.app.get(frame_bgr)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        return frame_bgr
