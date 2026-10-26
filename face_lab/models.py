"""Only prepare() may access the network. Inference only reads local files."""
import hashlib
import json
import urllib.request
from pathlib import Path

MODELS = {
    "yunet": dict(name="face_detection_yunet_2023mar.onnx", directory="face_detection_yunet",
                  sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", size=232589, license="MIT"),
    "sface": dict(name="face_recognition_sface_2021dec.onnx", directory="face_recognition_sface",
                  sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", size=38696353, license="Apache-2.0"),
}
for model in MODELS.values():
    model["url"] = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{model['directory']}/{model['name']}"
    model["license_url"] = f"https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/{model['directory']}/LICENSE"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, spec):
    if not path.is_file():
        raise ValueError(f"Missing model: {path}. Run prepare-models first.")
    if path.stat().st_size != spec["size"] or sha256(path) != spec["sha256"]:
        raise ValueError(f"Invalid model: {path}; expected size/SHA-256 mismatch (LFS pointer, HTML, corruption or other model).")


def prepare(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for spec in MODELS.values():
        target = folder / spec["name"]
        if target.exists():
            verify(target, spec)  # Never silently replace an existing file.
        else:
            temp = target.with_suffix(".download")
            if temp.exists():
                raise ValueError(f"Remove or inspect interrupted download first: {temp}")
            try:
                with urllib.request.urlopen(spec["url"], timeout=120) as r, temp.open("xb") as f:
                    while block := r.read(1024 * 1024):
                        f.write(block)
                verify(temp, spec)
                temp.rename(target)
            finally:
                temp.unlink(missing_ok=True)
        license_path = folder / (spec["directory"] + "_LICENSE.txt")
        if not license_path.exists():
            with urllib.request.urlopen(spec["license_url"], timeout=30) as r:
                license_path.write_bytes(r.read())
    (folder / "model_sources.json").write_text(json.dumps(MODELS, indent=2), encoding="utf-8")


def load(folder, score=0.9, nms=0.3, top_k=5000):
    import cv2
    import numpy as np
    folder = Path(folder)
    for spec in MODELS.values():
        verify(folder / spec["name"], spec)
    try:
        # Python handles Unicode Windows paths; OpenCV receives ONNX bytes, not narrow filenames.
        buffers = {key: np.frombuffer((folder / spec["name"]).read_bytes(), dtype=np.uint8) for key, spec in MODELS.items()}
        empty = np.empty(0, dtype=np.uint8)
        detector = cv2.FaceDetectorYN.create("onnx", buffers["yunet"], empty, (320, 320), score, nms, top_k,
                                             cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        recognizer = cv2.FaceRecognizerSF.create("onnx", buffers["sface"], empty,
                                                 cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        return detector, recognizer
    except (cv2.error, AttributeError) as e:
        raise ValueError(f"OpenCV YuNet/SFace loading failed; install requirements in a clean venv. {e}") from e
