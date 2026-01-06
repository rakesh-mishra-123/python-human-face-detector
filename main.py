import functions_framework
from google.cloud import storage
from face_detector import detect_faces
import logging

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@functions_framework.http
def detect_face(request):
    logger.info("Received face detection request.")
    request_json = request.get_json(silent=True)

    if not request_json:
        logger.error("Invalid JSON in request.")
        return {"status": "error", "message": "Invalid JSON"}, 400

    bucket_name = request_json.get("bucket_name")
    file_path = request_json.get("file_path")

    if not bucket_name:
        logger.error("bucket_name missing in request.")
        return {"status": "error", "message": "bucket_name missing"}, 400
    if not file_path:
        logger.error("file_path missing in request.")
        return {"status": "error", "message": "file_path missing"}, 400

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    if not blob.exists():
        logger.error(f"File not found: {file_path} in bucket {bucket_name}")
        return {"status": "error", "message": "File not found"}, 404

    image_bytes = blob.download_as_bytes()
    logger.info(f"Downloaded image bytes from {file_path}.")
    faces_count = detect_faces(image_bytes)
    logger.info(f"Faces detected: {faces_count}")

    return {
        "status": "success",
        "face_detected": faces_count > 0,
        "faces_count": faces_count
    }
