from insightface.app import FaceAnalysis
import numpy as np

class FaceDetector:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.app = FaceAnalysis(name="buffalo_l", providers=self._get_providers())
        self.app.prepare(ctx_id=self._get_ctx_id())

    def _get_ctx_id(self) -> int:
        return 0 if self.device == "cuda" else -1

    def _get_providers(self):
        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def detect(self, frame_bgr: np.ndarray):
        """
        Повертає список bounding box облич
        """
        faces = self.app.get(frame_bgr)
        bboxes = [face.bbox.astype(int).tolist() for face in faces]
        return bboxes
