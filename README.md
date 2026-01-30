# Edge Impulse Object Detection Demo (Arduino App Lab)

This project demonstrates real-time object detection using Edge Impulse models and a Flask web interface. Designed for Arduino App Lab and used for the MWC Barcelona 2026 workshops.

## Features

- Run Edge Impulse object detection models on images, videos, RTSP streams, or connected cameras
- Live web UI for viewing results and switching sources
- Easy setup for both Arduino App Lab and local development

![Demo overview](/docs/demo-overview-2.png)

## Quick Start (Arduino App Lab)

1. **Connect** your Arduino UNO Q.
2. **Clone this repo** into your ArduinoApps folder:
    ```bash
    cd ArduinoApps/
    git clone git@github.com:edgeimpulse/example-arduino-app-lab-object-detection-using-flask.git
    cd example-arduino-app-lab-object-detection-using-flask/
    ```
3. **Make the model executable** (or add your own):
    ```bash
    chmod +x models/rubber-ducky-fomo-linux-aarch64.eim
    ```
4. **Start the app:**
    ```bash
    arduino-app-cli app start .
    ```
5. **View logs (optional):**
    ```bash
    arduino-app-cli app logs .
    ```
6. **Stop the app:**
    ```bash
    arduino-app-cli app stop .
    ```

## Local Development

1. **Install Python 3.13.5** (recommended: [pyenv](https://github.com/pyenv/pyenv)):
    ```bash
    pyenv install 3.13.5
    pyenv local 3.13.5
    ```
2. **Set up a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3. **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r python/requirements.txt
    ```
4. **Run the app:**
    ```bash
    python python/main.py
    ```

## Usage

- Open your browser to the address shown in the terminal (default: http://localhost:5001)
- Use the web UI to select input source (image, video, RTSP, or camera)
- View original and detection results side by side

## Notes

- Place your images/videos in the `assets/` folder.
- Place your Edge Impulse `.eim` models in the `models/` folder.
- Only macOS (arm64) and Linux (aarch64) are supported for model inference.


