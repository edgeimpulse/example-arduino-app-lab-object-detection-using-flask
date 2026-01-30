# Edge Impulse Object Detection Demo (Arduino App Lab)

This project demonstrates real-time object detection using Edge Impulse models and a Flask web interface. Designed for Arduino App Lab and used for the MWC Barcelona 2026 workshops.

## Features

- Run Edge Impulse object detection models on images, videos, RTSP streams, or connected cameras
- Live web UI for viewing results and switching sources
- Easy setup for both Arduino App Lab and local development

![Demo overview](/docs/demo-overview-2.png)

## Quick Start (Arduino App Lab)

1. Open the **Arduino App Lab** and **Connect** your Arduino UNO Q.
2. SSH connexion: Click on the "terminal" icon on the bottom-left corner:
   
   ![App Lab - Open Terminal](/docs/app-lab-open-terminal.png)

3. Go to the `ArduinoApp/` directory:

```bash
cd home/arduino/ArduinoApps/
```

4. **Clone this repo** into your ArduinoApps folder:

    ```bash
    git clone https://github.com/edgeimpulse/example-arduino-app-lab-object-detection-using-flask.git
    cd example-arduino-app-lab-object-detection-using-flask/
    ```

5. In **Arduino App Lab**, go to **My Apps** (top-left corner). You should see the new application:

    ![App Lab - My Apps](/docs/arduino-app-lab-new-app.png)

6. **Make the models executable** (or add your own):

    ```bash
    chmod +x models/rubber-ducky-linux-aarch64.eim 
    chmod +x models/rubber-ducky-fomo-linux-aarch64.eim
    ```

7. **Start the app:**
   
    ```bash
    arduino-app-cli app start .
    ```

8. **View logs (optional):**
    ```bash
    arduino-app-cli app logs .
    ```

9.  **Stop the app:**
    ```bash
    arduino-app-cli app stop .
    ```

The logs will provide something similar to:

```
...
Edge Impulse runner initialized.
Server running at: http://172.18.0.2:5001
* Serving Flask app 'main'
* Debug mode: on
```

Note that this IP address is the internal docker network address. To get the local IP address of your UNO Q, use the following command:

```bash
hostname -I
192.168.1.8 172.17.0.1 172.18.0.1 2a01:e0a:c6:14b0:c428:1d78:47e0:1f8e 
```

1.  Open a web browser using your local IP address, here in this case `192.168.1.8:5001`:

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


