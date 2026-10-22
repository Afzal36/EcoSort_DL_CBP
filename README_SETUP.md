# EcoSort — Smart Waste Classification Using CNN

A beginner-friendly Streamlit interface for the six-class TrashNet CNN project.

## Features

- Upload JPG, JPEG, PNG, or WEBP waste images
- Predict one of six categories: cardboard, glass, metal, paper, plastic, trash
- Display prediction confidence and a probability chart
- Show general disposal guidance
- Train and evaluate the CNN locally

## 1. Open the project in VS Code

Extract this folder and open `EcoSort_VSCode` in VS Code.

Install Python 3.10 or 3.11 (recommended for TensorFlow compatibility).

## 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Download and arrange the TrashNet dataset

Download the TrashNet dataset used by the selected repository. Arrange images like this:

```text
EcoSort_VSCode/
├── app.py
├── train_model.py
├── requirements.txt
├── dataset-resized/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── models/
```

The training script scans image files inside each class folder. Folder names become the class labels. Do not rename the class folders after training.

## 4. Train the CNN and save the model

In the VS Code terminal:

```powershell
python train_model.py
```

Training can take some time depending on your CPU/GPU. It creates:

- `models/waste_classifier.keras`
- `models/class_names.json`
- `models/training_history.json`
- `models/test_metrics.json`

The model is trained locally from the dataset. It is not included in this starter bundle.

## 5. Run the interface

After training finishes:

```powershell
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open that address in your browser.

## Notes

- If your dataset folder has a different name, change `DATA_DIR` near the top of `train_model.py`.
- If the model reports low confidence, try a clear image of one object with good lighting.
- The output is a model prediction, not a guarantee that an item is recyclable. Follow local disposal rules.
- This is a starter implementation aligned to the selected repository's six classes and 128×128 CNN workflow. It retrains a compatible CNN using the repository's described architecture; it does not automatically import a trained model from the notebook.
