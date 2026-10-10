"""CricketLens - web app for identifying Indian cricketers from a photo.

Run:  python app.py      then open http://127.0.0.1:5000
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request

from faceid.core_names import pretty_name  # light import (no PyTorch)
from faceid.gallery import Gallery

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024  # several photos at once

METRICS = Path("models/metrics.json")
gallery = Gallery()
_recognizer = None


def get_recognizer():
    """Load the heavy models once, on first use."""
    global _recognizer
    if _recognizer is None:
        from faceid.recognizer import Recognizer

        _recognizer = Recognizer(gallery)
    return _recognizer


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/status")
def status():
    people = [{"id": k, "name": pretty_name(k), "count": v} for k, v in gallery.people().items()]
    metrics = json.loads(METRICS.read_text()) if METRICS.exists() else None
    return jsonify(ready=bool(people), people=people, metrics=metrics)


@app.post("/api/identify")
def identify():
    if not gallery.names:
        return jsonify(error="The gallery is empty. Add a player in the sidebar first."), 503
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Choose an image to upload."), 400
    try:
        return jsonify(get_recognizer().identify(file.stream))
    except Exception:
        app.logger.exception("identify failed")
        return jsonify(error="That file couldn't be read as an image. Try a JPG or PNG."), 422


@app.post("/api/teach")
def teach():
    name = request.form.get("name", "").strip()
    files = [f.stream for f in request.files.getlist("images") if f.filename]
    if not name or not files:
        return jsonify(error="Enter a name and choose at least one photo."), 400
    try:
        result = get_recognizer().teach(files, name, request.form.get("face", type=int))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception:
        app.logger.exception("teach failed")
        return jsonify(error="Couldn't learn from that photo."), 422
    if result["added"] == 0:
        return jsonify(error="No clear face found in the photo(s). Try a sharper, front-facing one."), 422
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Upload is too large. Try fewer or smaller photos."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
