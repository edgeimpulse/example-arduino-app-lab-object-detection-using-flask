import cv2
import os
import platform
import time
import logging
import socket
from flask import Flask, render_template, Response, request, jsonify

from utils.mock_dependencies import apply_mocks
apply_mocks()  # Apply mocks for six and pyaudio
from edge_impulse_linux.image import ImageImpulseRunner

app = Flask(__name__, static_folder='templates/assets')

# --- Configuration ---
SCALE_FACTOR = 6  # Scale factor for resizing inference frames
DESIRED_FPS = 30  # Target FPS for video processing

# --- Global variables ---
countObjects = 0
inferenceSpeed = 0
bounding_boxes = []
latest_high_res_frame = None
runner = None  # Global runner object
model_info = None  # Global model info

# Global variable to store the current source settings
current_source = {
    'source': 'image',
    'asset': 'rubber-duckies.jpg',
    'rtsp_url': ''
}

# Initialize the Edge Impulse runner when the app starts
def init_runner():
    global runner, MODEL_PATH, model_info

    system = platform.system().lower()
    machine = platform.machine().lower()

    print(f"Detected system: {system} {machine}")

    # Define the models directory relative to the script location
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.eim')]

    # Define model mapping for rubber-ducky based on OS/architecture
    model_mapping = {
        ('darwin', 'arm64'): "rubber-ducky-mac-arm64.eim",
        ('linux', 'aarch64'): "rubber-ducky-linux-aarch64.eim"
    }

    # Find the best model for this system
    model_name = None
    for (os_name, arch), model_file in model_mapping.items():
        if system == os_name and arch in machine:
            model_name = model_file
            break

    if not model_name:
        raise RuntimeError(f"Unsupported system: {system} {machine}. Only macOS (arm64) and Linux (aarch64) are supported.")

    model_path = os.path.join(models_dir, model_name)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Required model {model_name} not found in {models_dir}. Available models: {available_models}")

    MODEL_PATH = model_path  # Update the global MODEL_PATH
    print(f"Selected model: {model_name}")

    runner = ImageImpulseRunner(MODEL_PATH)
    model_info = runner.init()
    print(f"Model info: {model_info}")
    print("Edge Impulse runner initialized.")

def get_local_ip():
    """Get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"
    finally:
        s.close()
    return local_ip

# --- Video feed generator ---
def gen_video_frames():
    global latest_high_res_frame, current_source

    while True:
        if current_source['source'] == 'image':
            img_path = os.path.join(os.path.dirname(__file__), '..', 'assets', current_source['asset'])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found at: {img_path}")
            latest_high_res_frame = img.copy()
            ret, buffer = cv2.imencode('.jpg', latest_high_res_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)  # Add a small delay for images

        elif current_source['source'] == 'video':
            video_path = os.path.join(os.path.dirname(__file__), '..', 'assets', current_source['asset'])
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                latest_high_res_frame = frame.copy()
                ret, buffer = cv2.imencode('.jpg', latest_high_res_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.03)  # Adjust delay for video frame rate
            cap.release()

        elif current_source['source'] == 'rtsp':
            cap = cv2.VideoCapture(current_source['rtsp_url'])
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(current_source['rtsp_url'])
                    continue
                latest_high_res_frame = frame.copy()
                ret, buffer = cv2.imencode('.jpg', latest_high_res_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.03)  # Adjust delay for video frame rate
            cap.release()

# --- Inference and bounding box drawing ---
def gen_inference_frames():
    global countObjects, bounding_boxes, inferenceSpeed, latest_high_res_frame, runner

    while True:
        if latest_high_res_frame is None:
            time.sleep(0.1)
            continue

        img = cv2.cvtColor(latest_high_res_frame.copy(), cv2.COLOR_BGR2RGB)
        features, cropped = runner.get_features_from_image(img)
        res = runner.classify(features)

        if "result" in res:
            cropped = cv2.resize(cropped, (cropped.shape[1] * SCALE_FACTOR, cropped.shape[0] * SCALE_FACTOR))
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped = process_inference_result(res, cropped)

        ret, buffer = cv2.imencode('.jpg', cropped)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_inference_result(res, cropped):
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
                # Draw bounding boxes for object detection models
                if model_type == 'object_detection':
                    cropped = draw_bounding_box(cropped, bb)
                # Draw centroids for constrained object detection (FOMO) models
                elif model_type == 'constrained_object_detection':
                    cropped = draw_centroids(cropped, bb)
    return cropped

def draw_bounding_box(cropped, bb):
    """Draw bounding box for object detection models."""
    x = int(bb['x'] * SCALE_FACTOR)
    y = int(bb['y'] * SCALE_FACTOR)
    width = int(bb['width'] * SCALE_FACTOR)
    height = int(bb['height'] * SCALE_FACTOR)

    # Draw rectangle
    cropped = cv2.rectangle(cropped, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Add label
    label_text = f"{bb['label']}: {bb['value']:.2f}"
    label_position = (x, y - 10)
    cv2.putText(cropped, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return cropped

def draw_centroids(cropped, bb):
    """Draw centroids for constrained object detection (FOMO) models."""
    center_x = int((bb['x'] + bb['width'] / 2) * SCALE_FACTOR)
    center_y = int((bb['y'] + bb['height'] / 2) * SCALE_FACTOR)
    cropped = cv2.circle(cropped, (center_x, center_y), 10, (0, 255, 0), 2)

    # Add label
    label_text = f"{bb['label']}: {bb['value']:.2f}"
    label_position = (int(bb['x'] * SCALE_FACTOR), int(bb['y'] * SCALE_FACTOR) - 10)
    cv2.putText(cropped, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return cropped

# Add a new route to get available assets
@app.route('/get_assets', methods=['GET'])
def get_assets():
    asset_type = request.args.get('type', 'image')
    assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
    assets = []

    if os.path.exists(assets_dir):
        for file in os.listdir(assets_dir):
            if asset_type == 'image' and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                assets.append(file)
            elif asset_type == 'video' and file.lower().endswith(('.mp4', '.avi', '.mov')):
                assets.append(file)

    return jsonify(assets)

# Add a new route to handle setting changes
@app.route('/set_source', methods=['POST'])
def set_source():
    global current_source
    data = request.get_json()
    current_source['source'] = data.get('source', 'image')
    current_source['asset'] = data.get('asset', 'rubber-duckies.jpg')
    current_source['rtsp_url'] = data.get('rtspUrl', '')
    return jsonify({'status': 'success'})

# --- Flask routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/inference_feed')
def inference_feed():
    return Response(gen_inference_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/inference_speed')
def inference_speed():
    def get_inference_speed():
        while True:
            yield f"data:{inferenceSpeed}\n\n"
            time.sleep(0.1)
    return Response(get_inference_speed(), mimetype='text/event-stream')

@app.route('/object_counter')
def object_counter():
    def get_objects():
        while True:
            yield f"data:{countObjects}\n\n"
            time.sleep(0.1)
    return Response(get_objects(), mimetype='text/event-stream')

# --- Main ---
if __name__ == '__main__':
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    init_runner()  # Initialize the runner when the app starts
    local_ip = get_local_ip()
    print(f"Server running at: http://{local_ip}:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
