# Image Classification (Cats VS Dogs)

Cat vs Dog Image Classification using PyTorch & ResNet-18. Includes training notebook, inference script, Streamlit web app, saved model, evaluation metrics, ROC curve, confusion matrix, and demo screenshots. End-to-end project for training, testing, and deploying image classification models.

---

## Dataset

This project uses the **Cats vs Dogs dataset** originally provided by Microsoft and hosted on [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data).

### Access Instructions
1. Create a free Kaggle account if you don’t have one.
2. Go to the dataset page: https://www.kaggle.com/competitions/dogs-vs-cats/data
3. Accept the competition rules to unlock the dataset.
4. Download the ZIP file (`train.zip` and `test1.zip`).
5. Extract the contents and organize them as follows:
---

## Features
- **End-to-end pipeline**: From dataset preparation to deployment via a web app.
- **Cat vs Dog classification** using **ResNet-18** in PyTorch.
- **Evaluation metrics**: Accuracy, precision, recall, F1-score.
- **Visualizations**: Confusion matrix and ROC curve.
- **Saved trained model** for inference.
- **Inference script** to run batch predictions on test images.
- **Interactive Streamlit web app** for real-time image classification.
- **Demo screenshots** showcasing the web app interface.
- **Clean, modular repo structure** for easy reproducibility.

---

## Repository Structure
```
SafetyScorePrediction/
│── data/                           # Raw and processed datasets
│── notebooks/                      # Jupyter notebooks for experiments
│   └── train.ipynb                 # Model training notebook
│── scripts/                        # Source code for training/inference
│   ├── inference.py                # Inference script
│── app
|   ├── app.py                      # Streamlit web app
│── results/                        # Model outputs, metrics, logs                   
│   ├── metrics.json                # Evaluation metrics
│   └── figures/                    # Plots/graphs
│── models
|   ├── resnet18_catsdogs.pth       # Trained model
│── demo/                           # Screenshots of the webapp
│   ├── demo1.png
│   └── demo2.png
│── requirements.txt                # Python dependencies
│── .gitignore                      # Git ignore file
│── README.md                       # Project documentation
```

---

## Demo

<p align="center">
  <img src="demo/demo1.png" alt="Demo Screenshot 1" width="45%"/>
  <img src="demo/demo2.png" alt="Demo Screenshot 2" width="45%"/>
</p>

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sharad0x/ML_1_ImageClassification-Cats_VS_Dogs-_ARITHMATRIX
cd ML_1_ImageClassification-Cats_VS_Dogs-_ARITHMATRIX
pip install -r requirements.txt
```

---

##  Usage

Run the Streamlit app:

```bash
streamlit run app/app.py
```

---

## Results

Evaluation metrics are stored in `results/metrics.json`.  
The trained model is available in `models/resnet18_catsdogs.pth`.

---

## Requirements
See `requirements.txt` for the complete list of dependencies.

---

## Acknowledgements
- Built with **Python**, **scikit-learn**, **pandas**, **Streamlit**.
- Inspired by real-world safety prediction use cases.
