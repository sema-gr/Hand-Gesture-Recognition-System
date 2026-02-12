import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self, threshold=0.4):
        """
        threshold: максимальна відстань для визнання обличчя знайомим
        """
        self.threshold = threshold
        self.embeddings = []  # список np.ndarray (512,)
        self.user_ids = []    # список імен користувачів

    def add_user(self, user_id: str, embedding: np.ndarray):
        """
        Додати нового користувача в базу
        """
        self.user_ids.append(user_id)
        self.embeddings.append(embedding)

    def recognize(self, embedding: np.ndarray):
        """
        Порівняти embedding з базою, повернути user_id або None
        """
        if not self.embeddings:
            return None

        embeddings_matrix = np.vstack(self.embeddings)  # (N, 512)
        embedding = embedding.reshape(1, -1)

        # Косінусна схожість
        sims = cosine_similarity(embedding, embeddings_matrix)[0]
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]

        # Якщо схожість нижче threshold, користувач не знайдений
        if best_sim < (1 - self.threshold):
            return None

        return self.user_ids[best_idx]
