# CricketLens: Indian cricketer face identification

Upload a photo and CricketLens names the cricketer in it. Group photos work too: every face is
outlined and identified separately. Faces that don't match anyone are reported as "Not in the squad".

## How it works (dynamic, gallery-based)
1. **Detect:** MTCNN finds and aligns each face.
2. **Fingerprint:** FaceNet (pretrained on VGGFace2) turns each face into 512 numbers.
3. **Match:** the fingerprint is compared with a **gallery** of reference faces. The most similar player wins.
4. **Reject unknowns:** if the match is below the slider's threshold, the face is not named.

Because matching uses a gallery instead of a fixed trained classifier, **no retraining is needed to add a player**.
Add 3 to 10 photos of anyone and they are recognised immediately.

> No system can name a person it has no reference photo of. "Any cricketer" means "any cricketer in the gallery",
> so grow the gallery. The tools below make that fast.

## Quick start (Windows + VS Code)
1. Extract the zip. Open the folder in VS Code (*File -> Open Folder*, the folder that contains `app.py`).
2. Double-click `run.bat` (or follow "Manual setup"). First run takes several minutes.
3. Fill the gallery, using any of these:
   - **Automatic:** `python fetch_cricketers.py` downloads free-licensed photos of the ~45 players in `cricketers.txt`
     from Wikipedia/Wikimedia Commons (edit the list to add more), then `python build_gallery.py`.
   - **In the app:** sidebar -> *Add a player*: type a name and upload a few photos.
   - **By hand:** put photos in `dataset\<player_name>\` and run `python build_gallery.py`.
4. Open http://127.0.0.1:5000 and upload a photo.

### Teaching it
Under every result there is **Wrong person? Teach it** / **Teach it who this is**. Type the real name and the
face is added to the gallery, so the same mistake is much less likely next time.

## Manual setup (any OS)
```bash
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python fetch_cricketers.py      # optional: auto-download reference photos
python build_gallery.py
python app.py                   # open http://127.0.0.1:5000
```
Use Python 3.10, 3.11 or 3.12 (not 3.13).

## VS Code
Press **F5** and choose *run app*, *build gallery*, or *train model*.

## Optional: accuracy report for your project
`python train.py` trains and compares SVM / KNN / Logistic Regression on your dataset and writes
`models/metrics.json`. The app's **Model report** tab then shows accuracy, per-player precision/recall and a
confusion matrix. It needs 6+ photos per player. Great material for a write-up.

## Project layout
```
run.bat               one-click setup + launch (Windows)
app.py                Flask server (/api/status, /api/identify, /api/teach)
fetch_cricketers.py   downloads reference photos from Wikipedia / Commons (+ wrong-person filter)
cricketers.txt        the list of players to fetch
build_gallery.py      dataset/ -> models/gallery.npz
train.py              optional classifier comparison + accuracy report
faceid/               detection, embeddings, gallery, recognizer
static/               the web UI
dataset/              reference photos, one folder per player (the source of truth)
models/               generated files
```

## Limitations
- Accuracy depends on reference photo quality and variety. Helmets, sunglasses, blur and tiny faces are hard.
- Players with only 1-2 reference photos are matched less reliably. Add more through the app.
- Look-alike players are more likely to be confused. Use the slider and the Teach button.
- Photos from Wikimedia carry licences. Keep `dataset/*/SOURCES.txt` and credit them if you publish your work.
- Educational project: don't use it for surveillance or to identify private individuals.
