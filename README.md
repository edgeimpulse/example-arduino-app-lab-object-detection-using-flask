# Example Arduino App Lab object detection using Flask

This repository contains an example of a custom Arduino App Lab application running an Edge Impulse object detection model displaying the results on a WebUI.
This repository has been created for a workshop at the MWC Barcelona 2026.

## How to use

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
