import functions_framework
from google.cloud import storage
from face_detector import detect_faces

@functions_framework.http
def detect_face(request):
    request_json = request.get_json(silent=True)

    if not request_json:
        return {"status": "error", "message": "Invalid JSON"}, 400

    bucket_name = request_json.get("bucket_name")
    file_path = request_json.get("file_path")

    if not bucket_name:
        return {"status": "error", "message": "bucket_name missing"}, 400
    if not file_path:
        return {"status": "error", "message": "file_path missing"}, 400

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    if not blob.exists():
        return {"status": "error", "message": "File not found"}, 404

    image_bytes = blob.download_as_bytes()
    faces_count = detect_faces(image_bytes)

    return {
        "status": "success",
        "face_detected": faces_count > 0,
        "faces_count": faces_count
    }
