from ultralytics import YOLO

_model = None


def load_yolo_model():
    """Called once at FastAPI startup."""
    global _model
    _model = YOLO("app/models/best_yolo.pt")


def detect_objects(image_path: str, conf: float = 0.4) -> list[str]:
    """Runs inference and returns a list of unique detected class names."""
    if _model is None:
        raise RuntimeError("YOLO model not loaded. Call load_yolo_model() at startup.")

    results = _model.predict(image_path, conf=conf, verbose=False)
    labels = [_model.names[int(c)] for c in results[0].boxes.cls]
    return list(set(labels))
