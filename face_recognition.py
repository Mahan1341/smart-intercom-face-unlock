import cv2
import numpy as np
import onnxruntime as ort


class FaceRecognizer:
    def __init__(self, model_path: str, reference_image_path: str, threshold: float) -> None:
        self.threshold = threshold
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        reference_image = cv2.imread(reference_image_path)
        if reference_image is None:
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

        reference_face, _ = self.detect_face(reference_image)
        if reference_face is None:
            raise ValueError("No face detected in the reference image")

        self.reference_embedding = self.get_embedding(reference_face)

    def detect_face(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return None, None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return image[y : y + h, x : x + w], (x, y, w, h)

    def get_embedding(self, face: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            face,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        embedding = self.session.run(None, {self.input_name: blob})[0][0]
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Model returned a zero-norm embedding")
        return embedding / norm

    def compare(self, frame: np.ndarray):
        face, bbox = self.detect_face(frame)
        if face is None:
            return False, None, None

        embedding = self.get_embedding(face)
        similarity = float(np.dot(self.reference_embedding, embedding))
        return similarity >= self.threshold, similarity, bbox
