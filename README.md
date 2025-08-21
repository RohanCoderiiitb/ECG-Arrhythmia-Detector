# ❤️‍🩹 TinyML ECG Arrhythmia Detector (Second-Year Project) 🚀

> Real‑time ECG feature extraction on the host + on‑device inference with a TinyML model.

## 🌟 Overview
This repo implements a lightweight pipeline for arrhythmia detection using ECG:
- Python script to read ECG samples (from an AD8232 module via serial), filter the signal, detect R‑peaks, and extract **11 temporal features**.
- A TensorFlow Lite model converted to a C array for on‑device inference.

✅ **Model Accuracy**: Achieved **90.85%** accuracy during testing.

⚠️ **Status**: The project is still in progress. Deployment on **ESP32** has not been completed yet due to technical issues.

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

## 🧰 Challenges Faced
- ⚖️ **Class Imbalance**: Arrhythmia classes were underrepresented.  
  ✅ Mitigated using **SMOTE (Synthetic Minority Oversampling Technique)** and merging datasets.  
- ⚙️ **Deployment Issues**: ESP32 deployment is pending due to runtime/memory problems.

## 📸 Screenshots / Results
<img width="834" height="650" alt="Screenshot 2025-08-22 002947" src="https://github.com/user-attachments/assets/80a67282-3362-4464-942b-7eed40a96072" />  <img width="832" height="648" alt="Screenshot 2025-08-22 003008" src="https://github.com/user-attachments/assets/fa7516bf-fc92-4192-a53e-820df2112756" />  <img width="698" height="654" alt="Screenshot 2025-08-22 003030" src="https://github.com/user-attachments/assets/25428416-c2ac-4006-b593-683b3bdab738" />




## 🧾 License
MIT (or your preferred license).

---

> Made with ❤️ for TinyML exploration. Good luck this semester! 🎓
