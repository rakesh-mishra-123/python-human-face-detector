import cv2
import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROTO_PATH = "models/deploy.prototxt"
MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"

try:
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    logger.info("Loaded Caffe model and prototxt successfully.")
except Exception as e:
    logger.error(f"Error loading model files: {e}")
    raise

def detect_faces(image_bytes):
    logger.info("Starting face detection.")
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        logger.warning("Image decoding failed. Returning 0 faces.")
        return 0

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    faces = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        logger.debug(f"Detection {i}: confidence={confidence}")
        if confidence > 0.6:
            faces += 1

    logger.info(f"Detected {faces} faces.")
    return faces
