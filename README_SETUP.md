# EcoSort

EcoSort classifies waste images into cardboard, glass, metal, paper, plastic, or trash. The Streamlit app also provides MobileNetV2 feature-map inspection and class-specific Grad-CAM visualizations.

## Run locally

Use Python 3.12, which matches the tested TensorFlow environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Train or update the model

The training script expects the six class folders under `trashnet-dataset-main/`:

```text
trashnet-dataset-main/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Run `python train_model.py` from the project directory. Training requires the dataset and downloads MobileNetV2 ImageNet weights. It writes `models/waste_classifier.keras`, `models/class_names.json`, and `models/test_metrics.json`.

The deployed app requires the model and labels at `models/waste_classifier.keras` and `models/class_names.json`. These two files are intentionally included in Git; the dataset and generated metrics are ignored.

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository. Include `app.py`, `requirements.txt`, `.streamlit/config.toml`, `models/waste_classifier.keras`, and `models/class_names.json`. Do not add the dataset or `.venv`.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the main file.
3. Choose Python 3.12 in the app's advanced settings, then deploy. The cloud build installs the pinned packages from `requirements.txt`; no secrets are required.

The model file is about 24 MB, within GitHub's regular file-size limit. The app does not train during deployment. To publish a new model, retrain locally and update both the model and its matching class-label file.

## Notes

- The uploaded image limit is 10 MB.
- Prediction scores and Grad-CAM are model explanations, not guarantees about recyclability or disposal rules.
- Confirm local recycling guidance before sorting items.
