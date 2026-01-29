import cv2
import os
import platform
import time
import logging
from flask import Flask, render_template, Response
from edge_impulse_linux.image import ImageImpulseRunner
#from arduino.app_utils import App

app = Flask(__name__, static_folder='templates/assets')

# --- Configuration ---
VIDEO_PATH = "/assets/rubber-duckies.mp4"  # Path to your video file
MODEL_PATH = "models/rubber-ducky-mac-arm64.eim"
STREAM_URL = "rtsp://192.168.1.113:1935"
SCALE_FACTOR = 6  # Scale factor for resizing inference frames
DESIRED_FPS = 30  # Target FPS for video processing

# --- Global variables ---
countObjects = 0
inferenceSpeed = 0
bounding_boxes = []
latest_high_res_frame = None
runner = None  # Global runner object

# Initialize the Edge Impulse runner when the app starts
def init_runner():
    global runner, MODEL_PATH

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
    runner.init()
    print("Edge Impulse runner initialized.")

# --- Video feed generator ---
def gen_video_frames():
    global latest_high_res_frame
    img = cv2.imread("assets/rubber-duckies.jpg")  # Load the JPG image
    if img is None:
        raise FileNotFoundError("Image not found at: assets/rubber-duckies.jpg")

    while True:
        latest_high_res_frame = img.copy()  # Use the image as the frame
        ret, buffer = cv2.imencode('.jpg', latest_high_res_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1)  # Add a small delay to simulate a video feed

# --- Inference and bounding box drawing ---
def gen_inference_frames():
    global countObjects, bounding_boxes, inferenceSpeed, latest_high_res_frame, runner

    while True:
        if latest_high_res_frame is None:
            print("Waiting for video frame...")
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
    global countObjects, bounding_boxes, inferenceSpeed

    countObjects = 0
    bounding_boxes.clear()
    inferenceSpeed = res['timing']['classification']

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
                cropped = draw_centroids(cropped, bb)
    return cropped

def draw_centroids(cropped, bb):
    center_x = int((bb['x'] + bb['width'] / 2) * SCALE_FACTOR)
    center_y = int((bb['y'] + bb['height'] / 2) * SCALE_FACTOR)
    cropped = cv2.circle(cropped, (center_x, center_y), 10, (0, 255, 0), 2)
    label_text = f"{bb['label']}: {bb['value']:.2f}"
    label_position = (int(bb['x'] * SCALE_FACTOR), int(bb['y'] * SCALE_FACTOR) - 10)
    cv2.putText(cropped, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return cropped

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
    app.run(host="0.0.0.0", port=5001, debug=True)
