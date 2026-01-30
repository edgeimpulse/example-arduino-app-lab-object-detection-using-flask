
# --- Imports ---
import os
import cv2
import time
import socket
import logging
import platform
import requests
import io
import json
from flask import Flask, render_template, Response, request, jsonify

# --- App and Config ---
from utils.mock_dependencies import apply_mocks
apply_mocks()  # Apply mocks for six and pyaudio
from edge_impulse_linux.image import ImageImpulseRunner

app = Flask(__name__, static_folder='templates/assets')

# --- Constants ---
SCALE_FACTOR = 3  # Scale factor for resizing inference frames
MAX_CAMERAS = 5
MAX_RECONNECT_ATTEMPTS = 5
EI_INGEST_URL = "https://ingestion.edgeimpulse.com/api/"

# --- Globals ---
countObjects = 0
inferenceSpeed = 0
bounding_boxes = []
latest_high_res_frame = None
last_inference_frame_size = None  # (width, height) of the frame coordinates for bounding_boxes
runner = None
model_info = None

# Source management
current_source = {
    'source': 'image',
    'asset': 'rubber-duckies.jpg',
    'rtsp_url': '',
    'camera_index': 0,
    'connection_status': 'connecting'
}
source_change_requested = False
new_source_settings = None

from utils.mock_dependencies import apply_mocks
apply_mocks()  # Apply mocks for six and pyaudio
from edge_impulse_linux.image import ImageImpulseRunner

app = Flask(__name__, static_folder='templates/assets')

# --- Configuration ---
SCALE_FACTOR = 3  # Scale factor for resizing inference frames

# --- Global variables ---
countObjects = 0
inferenceSpeed = 0
bounding_boxes = []
latest_high_res_frame = None
runner = None
model_info = None

# Source management
current_source = {
    'source': 'image',
    'asset': 'rubber-duckies.jpg',
    'rtsp_url': '',
    'camera_index': 0,
    'connection_status': 'connecting'
}

source_change_requested = False
new_source_settings = None

def get_local_ip():
    """Get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

def get_compatible_models():
    """Return a list of model files compatible with the current system."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    all_models = [f for f in os.listdir(models_dir) if f.endswith('.eim')]
    compatible = []
    if system == 'darwin' and 'arm64' in machine:
        compatible = [f for f in all_models if 'mac-arm64' in f]
    elif system == 'linux' and 'aarch64' in machine:
        compatible = [f for f in all_models if 'linux-aarch64' in f]
    return compatible

def init_runner(model_name=None):
    """Initialize the Edge Impulse runner with the given model name (or default)."""
    global runner, MODEL_PATH, model_info, current_model_name
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    compatible_models = get_compatible_models()
    if not compatible_models:
        raise RuntimeError("No compatible models found for this system.")
    if model_name is None or model_name not in compatible_models:
        model_name = compatible_models[0]
    model_path = os.path.join(models_dir, model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model {model_name} not found in {models_dir}.")
    MODEL_PATH = model_path
    current_model_name = model_name
    print(f"Selected model: {model_name}")
    if runner:
        runner.stop()
    runner = ImageImpulseRunner(MODEL_PATH)
    model_info = runner.init()
    print(f"Model info: {model_info}")
    print("Edge Impulse runner initialized.")

# Track current model name
current_model_name = None
# Endpoint to get available models for dropdown
@app.route('/get_models')
def get_models():
    """Return compatible model filenames for dropdown."""
    return jsonify(get_compatible_models())

def gen_video_frames():
    """Generate video frames from the selected source."""
    global latest_high_res_frame, current_source, source_change_requested, new_source_settings
    cap = None
    reconnect_attempts = 0
    while True:
        if source_change_requested:
            source_change_requested = False
            reconnect_attempts = 0
            if cap is not None:
                cap.release()
                time.sleep(0.5)
            current_source.update(new_source_settings)
            current_source['connection_status'] = 'connecting'
            print(f"Switched to source: {current_source}")
        try:
            src = current_source['source']
            if src == 'camera':
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(current_source['camera_index'])
                    if not cap.isOpened():
                        print(f"Failed to open camera {current_source['camera_index']}")
                        time.sleep(1)
                        continue
                success, frame = cap.read()
                if not success:
                    print("Failed to read frame from camera")
                    cap.release()
                    time.sleep(1)
                    continue
                latest_high_res_frame = frame.copy()
                current_source['connection_status'] = 'connected'
                ret, buffer = cv2.imencode('.jpg', latest_high_res_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.03)
            elif src == 'image':
                img_path = os.path.join(os.path.dirname(__file__), '..', 'assets', current_source['asset'])
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Image not found: {img_path}")
                    time.sleep(1)
                    continue
                latest_high_res_frame = img.copy()
                current_source['connection_status'] = 'connected'
                ret, buffer = cv2.imencode('.jpg', latest_high_res_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(1)
            elif src == 'video':
                if cap is None or not cap.isOpened():
                    video_path = os.path.join(os.path.dirname(__file__), '..', 'assets', current_source['asset'])
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Failed to open video: {video_path}")
                        time.sleep(1)
                        continue
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.1)
                    continue
                latest_high_res_frame = frame.copy()
                current_source['connection_status'] = 'connected'
                ret, buffer = cv2.imencode('.jpg', latest_high_res_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.03)
            elif src == 'rtsp':
                if cap is None or not cap.isOpened():
                    rtsp_url = current_source['rtsp_url'].rstrip('/')
                    print(f"Attempting to connect to RTSP: {rtsp_url}")
                    cap = cv2.VideoCapture()
                    if not cap.open(rtsp_url, cv2.CAP_FFMPEG):
                        print(f"Failed to open RTSP stream: {rtsp_url}")
                        current_source['connection_status'] = 'disconnected'
                        time.sleep(1)
                        continue
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    current_source['connection_status'] = 'connected'
                success, frame = cap.read()
                if not success:
                    reconnect_attempts += 1
                    print(f"Failed to read frame from RTSP, reconnect attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}")
                    current_source['connection_status'] = 'reconnecting'
                    if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                        print("Max reconnect attempts reached, giving up for now")
                        current_source['connection_status'] = 'disconnected'
                        cap.release()
                        time.sleep(5)
                        reconnect_attempts = 0
                        continue
                    cap.release()
                    time.sleep(1)
                    continue
                reconnect_attempts = 0
                latest_high_res_frame = frame.copy()
                ret, buffer = cv2.imencode('.jpg', latest_high_res_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.03)
        except Exception as e:
            print(f"Error in video feed: {str(e)}")
            current_source['connection_status'] = 'disconnected'
            if cap and cap.isOpened():
                cap.release()
            time.sleep(1)

def gen_inference_frames():
    """Generate inference frames with object detection."""
    global countObjects, bounding_boxes, inferenceSpeed, latest_high_res_frame, runner, last_inference_frame_size

    while True:
        if latest_high_res_frame is None:
            time.sleep(0.1)
            continue

        try:
            img = cv2.cvtColor(latest_high_res_frame.copy(), cv2.COLOR_BGR2RGB)

            # Try to get features with error handling
            try:
                features, cropped = runner.get_features_from_image(img)
                # Store the coordinate space of the returned bounding boxes (pre SCALE_FACTOR rendering)
                last_inference_frame_size = (int(cropped.shape[1]), int(cropped.shape[0]))
                res = runner.classify(features)
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                time.sleep(0.1)
                continue

            if "result" in res:
                cropped = cv2.resize(cropped, (cropped.shape[1] * SCALE_FACTOR, cropped.shape[0] * SCALE_FACTOR))
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                cropped = process_inference_result(res, cropped)

            ret, buffer = cv2.imencode('.jpg', cropped)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error during inference: {str(e)}")
            time.sleep(0.1)

def process_inference_result(res, cropped):
    """Process inference results and draw bounding boxes/centroids."""
    global countObjects, bounding_boxes, inferenceSpeed, model_info
    countObjects = 0
    bounding_boxes.clear()
    inferenceSpeed = res['timing']['classification']
    model_type = model_info['model_parameters']['model_type']

    if "bounding_boxes" in res["result"]:
        for bb in res["result"]["bounding_boxes"]:
            if bb['value'] > 0:
                countObjects += 1
                bounding_boxes.append({
                    'label': bb['label'],
                    'x': int(bb['x']),
                    'y': int(bb['y']),
                    'width': int(bb['width']),
                    'height': int(bb['height']),
                    'confidence': bb['value']
                })
                if model_type == 'object_detection':
                    cropped = draw_bounding_box(cropped, bb)
                elif model_type == 'constrained_object_detection':
                    cropped = draw_centroids(cropped, bb)
    return cropped

def draw_bounding_box(cropped, bb):
    """Draw bounding box for object detection models."""
    x = int(bb['x'] * SCALE_FACTOR)
    y = int(bb['y'] * SCALE_FACTOR)
    width = int(bb['width'] * SCALE_FACTOR)
    height = int(bb['height'] * SCALE_FACTOR)
    cv2.rectangle(cropped, (x, y), (x + width, y + height), (0, 255, 0), 2)
    label_text = f"{bb['label']}: {bb['value']:.2f}"
    cv2.putText(cropped, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return cropped

def draw_centroids(cropped, bb):
    """Draw centroids for FOMO models."""
    center_x = int((bb['x'] + bb['width'] / 2) * SCALE_FACTOR)
    center_y = int((bb['y'] + bb['height'] / 2) * SCALE_FACTOR)
    cv2.circle(cropped, (center_x, center_y), 10, (0, 255, 0), 2)
    label_text = f"{bb['label']}: {bb['value']:.2f}"
    cv2.putText(cropped, label_text, (int(bb['x'] * SCALE_FACTOR), int(bb['y'] * SCALE_FACTOR) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return cropped

@app.route('/get_assets')
def get_assets():
    """Get available images/videos from assets folder - restored working version"""
    asset_type = request.args.get('type', 'image')
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))

    print(f"\n=== DEBUG INFO ===")
    print(f"Looking for {asset_type} in: {assets_dir}")
    print(f"Directory exists: {os.path.exists(assets_dir)}")

    assets = []
    if os.path.exists(assets_dir):
        print("Files in directory:")
        for file in os.listdir(assets_dir):
            file_path = os.path.join(assets_dir, file)
            if os.path.isfile(file_path):
                print(f"  - {file}")
                if asset_type == 'image' and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    assets.append(file)
                elif asset_type == 'video' and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    assets.append(file)

    print(f"Found {len(assets)} {asset_type}s: {assets}")
    return jsonify(assets)

@app.route('/get_cameras', methods=['GET'])
def get_cameras():
    """Get available connected cameras"""
    cameras = []
    for i in range(MAX_CAMERAS):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return jsonify(cameras)

@app.route('/set_source', methods=['POST'])
def set_source():
    global source_change_requested, new_source_settings, current_model_name
    data = request.get_json()
    requested_model = data.get('modelName')
    # If model changed, re-init runner
    if requested_model and requested_model != current_model_name:
        try:
            init_runner(requested_model)
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    new_source_settings = {
        'source': data.get('source', 'image'),
        'asset': data.get('asset', 'rubber-duckies.jpg'),
        'rtsp_url': data.get('rtspUrl', ''),
        'camera_index': data.get('cameraIndex', 0)
    }
    source_change_requested = True
    return jsonify({'status': 'success'})

@app.route('/get_connection_status')
def get_connection_status():
    """Get the current connection status."""
    return jsonify({
        'status': current_source['connection_status'],
        'source': current_source['source']
    })

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video feed endpoint."""
    return Response(gen_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/inference_feed')
def inference_feed():
    """Inference feed endpoint."""
    return Response(gen_inference_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/inference_speed')
def inference_speed():
    """Stream inference speed."""
    def generate():
        while True:
            yield f"data:{inferenceSpeed}\n\n"
            time.sleep(0.1)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/object_counter')
def object_counter():
    """Stream object count."""
    def generate():
        while True:
            yield f"data:{countObjects}\n\n"
            time.sleep(0.1)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/upload_edge_impulse', methods=['POST'])
def upload_edge_impulse():
    data = request.get_json() or {}
    api_key = data.get('apiKey')
    category = data.get('category', 'training')
    include_labels = bool(data.get('includeLabels', False))
    if not api_key:
        return jsonify({'status': 'error', 'message': 'Missing API key'}), 400

    if category not in ['training', 'testing', 'split']:
        return jsonify({'status': 'error', 'message': 'Invalid category'}), 400

    if latest_high_res_frame is None:
        return jsonify({'status': 'error', 'message': 'No frame available yet. Wait for the Original feed to load.'}), 409

    try:
        # Encode the latest frame as JPEG
        ok, buffer = cv2.imencode('.jpg', latest_high_res_frame)
        if not ok:
            return jsonify({'status': 'error', 'message': 'Failed to encode JPEG'}), 500

        filename = f"original-{int(time.time())}.jpg"
        files = {'data': (filename, io.BytesIO(buffer.tobytes()), 'image/jpeg')}
        headers = {'x-api-key': api_key}

        if include_labels:
            # Scale bounding boxes from inference/cropped space to original frame space
            boxes_for_upload = []
            orig_h, orig_w = int(latest_high_res_frame.shape[0]), int(latest_high_res_frame.shape[1])
            inf_w, inf_h = (last_inference_frame_size or (orig_w, orig_h))

            # Guard against division by zero
            if inf_w <= 0 or inf_h <= 0:
                inf_w, inf_h = orig_w, orig_h

            scale_x = orig_w / inf_w
            scale_y = orig_h / inf_h

            for bb in (bounding_boxes or []):
                x = int(round(int(bb.get('x', 0)) * scale_x))
                y = int(round(int(bb.get('y', 0)) * scale_y))
                w = int(round(int(bb.get('width', 0)) * scale_x))
                h = int(round(int(bb.get('height', 0)) * scale_y))
                boxes_for_upload.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'label': str(bb.get('label', ''))
                })

            headers['x-bounding-boxes'] = json.dumps(boxes_for_upload)
        url = EI_INGEST_URL + f"{category}/files"
        resp = requests.post(url, files=files, headers=headers, params={'filename': filename}, timeout=30)

        if 200 <= resp.status_code < 300:
            return jsonify({'status': 'success'})

        return jsonify({'status': 'error', 'message': resp.text, 'status_code': resp.status_code}), resp.status_code
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    init_runner()
    local_ip = get_local_ip()
    print(f"Server running at: http://{local_ip}:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
    