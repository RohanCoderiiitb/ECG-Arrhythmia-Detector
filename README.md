# ❤️‍🩹 TinyML ECG Arrhythmia Detector (Second-Year Project) 🚀

> Real‑time ECG feature extraction on the host + on‑device inference with a TinyML model.

## 🌟 Overview
This repo implements a lightweight pipeline for arrhythmia detection using ECG:
- Python script to read ECG samples (from an AD8232 module via serial), filter the signal, detect R‑peaks, and extract **11 temporal features**.
- A TensorFlow Lite model converted to a C array for on‑device inference.

## 🧠 Model (TFLite → C array)
The embedded model is exposed as `ecg_arrhythmia_model_tflite` in `model.h` (byte array, with a reported length constant of `ecg_arrhythmia_model_tflite_len = 37744`). Use it with TFLite Micro or your preferred inference runtime on microcontrollers.  

**Files**
- `model.h` — C array with the TFLite flatbuffer.
- `ecg-arrhythmia-model.tflite` — original TFLite model.

## 🧪 Feature Extraction Pipeline
Main steps handled in `feature_extractor.py`:
1. **Serial read** from the ECG board (default: `COM4`) at 50 Hz for a set duration.
2. **Band‑pass filter** (nominal docstring 0.5–40 Hz, current parameters clamp to 0.5–20 Hz; adjust as needed).
3. **R‑peak detection** using moving average over squared, normalized ECG.
4. **11 features** (per beat) computed from peak timings and intervals:  
   `['0_pre-RR','0_post-RR','0_pPeak','0_tPeak','0_rPeak','0_sPeak','0_qPeak','0_qrs_interval','0_pq_interval','0_qt_interval','0_st_interval']`

## 🗂️ Project Structure
```
.
├── feature_extractor.py        # Read serial ECG, filter, detect R-peaks, extract 11 features
├── model.h                     # TFLite model as C array for MCU inference
├── ecg-arrhythmia-model.tflite # Original model (for reference)
├── model.h                     # (same as above; kept for MCU builds)
└── Neural_Network.ipynb        # Training notebook (reference)
```

## ⚙️ Setup (Python)
```bash
python -m venv .venv
# Activate: Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install numpy scipy matplotlib pyserial
```

## ▶️ Run (Feature Extraction Demo)
Update the serial port if needed (e.g., `COM4` → `COM3` on Windows or `/dev/ttyUSB0` on Linux), then:
```bash
python feature_extractor.py
```
The script prints the 11‑element `float32` feature vector and can plot raw vs. filtered ECG.

## 🔌 MCU Inference (sketch)
- Include `model.h` in your firmware.
- Initialize a TFLite Micro interpreter with the model buffer.
- Feed the 11‑D feature vector as input and read the output logits/probabilities.

> Tip: ensure the **feature order and scaling** on‑device exactly match the Python extractor.

## 🧰 Notes & Next Steps
- Align the filter passband with your sensor/noise profile.
- Validate R‑peaks visually (matplotlib) to ensure beat alignment.
- Consider buffering multiple beats and majority voting to stabilize predictions.
- Add a simple UART protocol to stream features to the MCU if you keep extraction on host first.

## 🧾 License
MIT (or your preferred license).

---

# 🧩 “Code to write README on GitHub”

## Option A — Quick Git commands (local → GitHub)
```bash
# 1) Create README.md (if you haven't already)
echo "# TinyML ECG Arrhythmia Detector" > README.md

# 2) Initialize repo (if new) and push
git init
git add README.md feature_extractor.py model.h ecg-arrhythmia-model.tflite Neural_Network.ipynb
git commit -m "Add README and project files"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Option B — Python script that writes README.md and commits
```python
# save as write_readme_and_commit.py
from pathlib import Path
import subprocess

README = Path("README.md")
README.write_text("""# TinyML ECG Arrhythmia Detector

See sections for setup, pipeline, and MCU inference.
""", encoding="utf-8")

subprocess.run(["git", "init"], check=True)
subprocess.run(["git", "add", "README.md"], check=True)
subprocess.run(["git", "commit", "-m", "Add README"], check=True)
# set your remote before pushing:
# subprocess.run(["git", "remote", "add", "origin", "https://github.com/<user>/<repo>.git"], check=True)
# subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
```

## Option C — GitHub Actions (auto‑check README presence)
Create `.github/workflows/verify-readme.yml`:
```yaml
name: Verify README
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Ensure README exists
        run: |
          test -f README.md || (echo "README.md missing" && exit 1)
```

---
