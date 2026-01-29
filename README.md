# Example Arduino App Lab object detection using Flask

This repository contains an example of a custom Arduino App Lab application running an Edge Impulse object detection model displaying the results on a WebUI.
This repository has been created for a workshop at the MWC Barcelona 2026.

## How to use

Connect to your Arduino UNO Q

Then navigate to the `ArduinoApps/` folder and clone this repository:

```bash
cd ArduinoApps/
git clone git@github.com:edgeimpulse/example-arduino-app-lab-object-detection-using-flask.git
cd example-arduino-app-lab-object-detection-using-flask/
```

Make the default model executable (or add your own):

```bash
chmod +x models/rubber-ducky-fomo-linux-aarch64.eim
```

Run the app:

```bash
arduino-app-cli app start .
```

If you want to see the logs:

```bash
arduino-app-cli app logs .
```

To stop the app:

```
arduino-app-cli app stop .
```

## Development

The Arduino App Lab uses python 3.13.5, if you want to develop on another computer, let's set the envrionment:

Install Python 3.13.5:

```
pyenv install 3.13.5
```

```
pyenv local 3.13.5
```

Setup the Virtual Environment:

```
python -m venv .venv
```

Activate the venv:

```
source .venv/bin/activate
```

Install the dependencies:
```
pip install --upgrade pip
pip install -r python/requirements.txt
```

Run the app:
```
python python/main.py
```
