# RecycleVision – Garbage Image Classification Using Deep Learning

## Project Overview

RecycleVision is a deep learning project that classifies garbage images into 12 categories to support waste identification and recycling.

## Objectives

* Classify garbage images into 12 categories.
* Apply image preprocessing and exploratory data analysis (EDA).
* Compare a baseline CNN with transfer learning models.
* Evaluate model performance using accuracy, precision, recall, and F1-score.
* Provide a user-friendly web application for image prediction.

## Garbage Categories

Battery, Biological, Brown Glass, Cardboard, Clothes, Green Glass, Metal, Paper, Plastic, Shoes, Trash, and White Glass.

## Models Used

* Baseline CNN
* MobileNetV2
* ResNet50
* EfficientNetB0

EfficientNetB0 was selected as the best-performing model based on the test-set evaluation.

## Technologies

* Python
* TensorFlow / Keras
* Pandas and NumPy
* Matplotlib
* Scikit-learn
* Streamlit

## Project Structure

* `app.py` – Streamlit web application
* `src/` – Dataset preparation, preprocessing, model training, and evaluation scripts
* `models/` – Saved trained model
* `reports/` – EDA outputs, evaluation reports, and model comparison
* `requirements.txt` – Python dependencies

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application from the project root:

```bash
streamlit run app.py
```

Upload a garbage image to view the predicted category, confidence score, and top three predictions.

## Evaluation

The models were evaluated on a held-out test dataset using accuracy, macro precision, macro recall, macro F1-score, and weighted F1-score. Refer to `reports/model_comparison.csv` for the comparison results.

## Conclusion

RecycleVision demonstrates how deep learning and transfer learning can be applied to garbage image classification through a web-based prediction interface.

