from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image


PROJECT_PATH = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_PATH / "models" / "efficientnetb0_best.keras"
TRAIN_MANIFEST_PATH = PROJECT_PATH / "reports" / "train_manifest.csv"
COMPARISON_PATH = PROJECT_PATH / "reports" / "model_comparison.csv"

IMAGE_SIZE = (224, 224)


CLASS_DISPLAY_NAMES = {
    "battery": "Battery",
    "biological": "Biological",
    "brown-glass": "Brown Glass",
    "cardboard": "Cardboard",
    "clothes": "Clothes",
    "green-glass": "Green Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "shoes": "Shoes",
    "trash": "Trash",
    "white-glass": "White Glass",
}


st.set_page_config(
    page_title="RecycleVision",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 19px;
        margin-bottom: 25px;
        color: #64748b;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 15px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 12px;
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(40, 167, 69, 0.10);
        border: 1px solid rgba(40, 167, 69, 0.35);
        margin: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


@st.cache_data
def load_class_names():
    if not TRAIN_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {TRAIN_MANIFEST_PATH}"
        )

    df = pd.read_csv(TRAIN_MANIFEST_PATH)

    if "class_name" not in df.columns:
        raise ValueError(
            "The training manifest must contain a class_name column."
        )

    class_names = sorted(
        df["class_name"].dropna().unique().tolist()
    )

    if len(class_names) != 12:
        raise ValueError(
            f"Expected 12 classes, but found {len(class_names)}."
        )

    return class_names


def display_class_name(class_name):
    return CLASS_DISPLAY_NAMES.get(
        class_name,
        class_name.replace("-", " ").title(),
    )


def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_array


def predict_image(image, model, class_names):
    image_array = prepare_image(image)

    predictions = model.predict(
        image_array,
        verbose=0,
    )[0]

    if len(predictions) != len(class_names):
        raise ValueError(
            "The model output does not match the number of classes."
        )

    top_indices = np.argsort(predictions)[::-1][:3]

    top_predictions = pd.DataFrame(
        {
            "Category": [
                display_class_name(class_names[index])
                for index in top_indices
            ],
            "Confidence (%)": [
                float(predictions[index]) * 100
                for index in top_indices
            ],
        }
    )

    predicted_class = class_names[int(np.argmax(predictions))]
    confidence = float(np.max(predictions)) * 100

    return predicted_class, confidence, top_predictions


def introduction_page():
    st.markdown(
        '<div class="main-title">♻️ RecycleVision</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Garbage Image Classification Using Deep Learning'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        RecycleVision is a deep learning application that classifies
        garbage images into different categories.

        Users can upload an image and obtain a predicted garbage
        category using a trained EfficientNetB0 image classification model.
        """
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True,
    )

    overview_col1, overview_col2 = st.columns(2)

    with overview_col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>🎯 Project Objectives</h3>
                <ul>
                    <li>Automatically classify garbage images.</li>
                    <li>Reduce manual image identification.</li>
                    <li>Compare deep learning classification models.</li>
                    <li>Build an interactive image prediction application.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with overview_col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>⚙️ Project Methodology</h3>
                <ol>
                    <li>Inspect and clean the dataset.</li>
                    <li>Remove duplicate images.</li>
                    <li>Split images into training, validation and test sets.</li>
                    <li>Train transfer-learning models.</li>
                    <li>Evaluate the models and select the best performer.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        '<div class="section-title">Dataset Information</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Original Images",
            "15,515",
        )

    with metric2:
        st.metric(
            "Garbage Categories",
            "12",
        )

    with metric3:
        st.metric(
            "Selected Model",
            "EfficientNetB0",
        )

    st.caption(
        "The original dataset contained 15,515 images. "
        "Duplicate handling and dataset splitting were performed "
        "during preprocessing."
    )

    st.markdown(
        '<div class="section-title">Garbage Categories</div>',
        unsafe_allow_html=True,
    )

    class_names = [
        "Battery",
        "Biological",
        "Brown Glass",
        "Cardboard",
        "Clothes",
        "Green Glass",
        "Metal",
        "Paper",
        "Plastic",
        "Shoes",
        "Trash",
        "White Glass",
    ]

    category_columns = st.columns(4)

    for index, category in enumerate(class_names):
        with category_columns[index % 4]:
            st.info(category)

    st.divider()

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Three transfer-learning models were trained and evaluated
        on the same test dataset.
        """
    )

    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH)

        required_columns = {
            "model",
            "test_accuracy",
            "macro_f1_score",
        }

        if required_columns.issubset(comparison.columns):
            display_comparison = comparison[
                [
                    "model",
                    "test_accuracy",
                    "macro_f1_score",
                ]
            ].copy()

            display_comparison["test_accuracy"] = (
                display_comparison["test_accuracy"] * 100
            ).round(2)

            display_comparison["macro_f1_score"] = (
                display_comparison["macro_f1_score"] * 100
            ).round(2)

            display_comparison = display_comparison.rename(
                columns={
                    "model": "Model",
                    "test_accuracy": "Test Accuracy (%)",
                    "macro_f1_score": "Macro F1-score (%)",
                }
            )

            st.dataframe(
                display_comparison,
                hide_index=True,
                use_container_width=True,
            )

    st.success(
        "EfficientNetB0 was selected because it achieved "
        "96.56% test accuracy and a 95.00% macro F1-score "
        "in the recorded evaluation."
    )

    st.divider()

    st.markdown(
        """
        **Technology Stack:** Python · TensorFlow · Keras ·
        NumPy · Pandas · Streamlit
        """
    )

    st.caption(
        "RecycleVision | Garbage Image Classification Using Deep Learning"
    )


def prediction_page():
    st.markdown(
        '<div class="main-title">🔍 Garbage Prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload an image to identify its garbage category'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Select an image of garbage, review the image preview,
        and click **Predict** to obtain the model's classification.
        """
    )

    try:
        model = load_model()
        class_names = load_class_names()

    except Exception as error:
        st.error(
            f"Unable to load the model or class names: {error}"
        )
        return

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG and WEBP.",
        key="garbage_image_upload",
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to get started."
        )

        st.markdown(
            """
            <div class="info-card">
                <h4>How to use</h4>
                <ol>
                    <li>Choose a garbage image from your computer.</li>
                    <li>Check the image preview.</li>
                    <li>Click the Predict button.</li>
                    <li>View the predicted category and confidence scores.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    try:
        image = Image.open(uploaded_file).convert("RGB")

    except Exception:
        st.error(
            "The uploaded file could not be opened as an image."
        )
        return

    current_upload_id = (
        uploaded_file.name,
        uploaded_file.size,
    )

    if st.session_state.get("upload_id") != current_upload_id:
        st.session_state["upload_id"] = current_upload_id
        st.session_state.pop("prediction_result", None)

    st.divider()

    st.markdown(
        '<div class="section-title">Uploaded Image</div>',
        unsafe_allow_html=True,
    )

    image_col, details_col = st.columns([1, 1])

    with image_col:
        st.image(
            image,
            caption="Image selected for prediction",
            use_container_width=True,
        )

    with details_col:
        st.subheader("Image Details")

        st.write(
            f"**File name:** {uploaded_file.name}"
        )

        st.write(
            f"**Image dimensions:** {image.width} × {image.height}"
        )

        st.write(
            f"**Format:** {uploaded_file.type}"
        )

        predict_button = st.button(
            "🔎 Predict",
            type="primary",
            use_container_width=True,
        )

        if predict_button:
            try:
                with st.spinner("Analyzing the image..."):
                    predicted_class, confidence, top_predictions = (
                        predict_image(
                            image,
                            model,
                            class_names,
                        )
                    )

                st.session_state["prediction_result"] = {
                    "upload_id": current_upload_id,
                    "class": predicted_class,
                    "confidence": confidence,
                    "top_predictions": top_predictions,
                }

            except Exception as error:
                st.error(
                    f"Prediction failed: {error}"
                )
                return

    result = st.session_state.get("prediction_result")

    if result and result["upload_id"] == current_upload_id:
        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True,
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.success(
                "Predicted Category: "
                + display_class_name(result["class"])
            )

        with result_col2:
            st.metric(
                "Model Confidence",
                f"{result['confidence']:.2f}%",
            )

        st.markdown(
            '<div class="section-title">Top 3 Predictions</div>',
            unsafe_allow_html=True,
        )

        top_predictions = result["top_predictions"].copy()

        top_predictions["Confidence (%)"] = (
            top_predictions["Confidence (%)"].round(2)
        )

        st.dataframe(
            top_predictions,
            hide_index=True,
            use_container_width=True,
        )

        chart_data = top_predictions.set_index("Category")

        st.bar_chart(
            chart_data["Confidence (%)"]
        )

        st.caption(
            "Confidence represents the model's predicted probability. "
            "It does not guarantee that the classification is correct."
        )

    st.divider()

    st.caption(
        "RecycleVision | Powered by EfficientNetB0"
    )


def main():
    st.sidebar.title("♻️ RecycleVision")

    st.sidebar.write(
        "Garbage Image Classification"
    )

    selected_page = st.sidebar.radio(
        "Navigation",
        [
            "Introduction",
            "Garbage Prediction",
        ],
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        """
        **About the Project**

        A deep learning application for classifying
        garbage images into 12 categories.
        """
    )

    st.sidebar.caption(
        "Model: EfficientNetB0"
    )

    if selected_page == "Introduction":
        introduction_page()

    elif selected_page == "Garbage Prediction":
        prediction_page()


if __name__ == "__main__":
    main()