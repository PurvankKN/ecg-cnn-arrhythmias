
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="ECG CNN Arrhythmia Diagnosis",
    page_icon="💗",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "ecg_test_parts"

# --------------------------------------------------
# LOAD MODEL AND METADATA
# --------------------------------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(
        BASE_DIR / "ecg_cnn_model.keras"
    )

    with open(BASE_DIR / "ecg_label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open(BASE_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    # Supports the dictionary format used by the training notebook.
    if isinstance(encoder, dict):
        classes = encoder.get("class_names")
        if classes is None:
            raise ValueError(
                "The label encoder dictionary has no 'class_names' key."
            )
    else:
        classes = list(encoder.classes_)

    return model, [str(c) for c in classes], metadata


st.title("💗 Explainable AI Framework for ECG-Based Arrhythmia Diagnosis")
st.caption(
    "1D CNN · MIT-BIH Arrhythmia Database · Educational research prototype"
)
st.warning(
    "For education and research only. This is not a medical device."
)

try:
    model, classes, metadata = load_assets()
except Exception as e:
    st.error(f"Could not load the trained CNN assets: {e}")
    st.stop()

# --------------------------------------------------
# LOAD AND COMBINE CSV PARTS
# --------------------------------------------------
st.header("ECG Test Dataset")

if not DATA_DIR.exists():
    st.error(
        "The folder 'ecg_test_parts' was not found in the repository."
    )
    st.stop()

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    st.error(
        "No CSV files were found in ecg_test_parts. "
        "Upload the CSV parts into that folder on GitHub."
    )
    st.stop()

st.write(f"Found **{len(csv_files)} CSV part(s)**.")

try:
    frames = []
    for file in csv_files:
        part = pd.read_csv(file)

        if "label" not in part.columns:
            st.error(f"Missing 'label' column in {file.name}.")
            st.stop()

        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
except Exception as e:
    st.error(f"Could not read the CSV parts: {e}")
    st.stop()

if df.empty:
    st.error("The combined dataset is empty.")
    st.stop()

if df.columns.tolist().count("label") != 1:
    st.error("The dataset must have exactly one 'label' column.")
    st.stop()

st.success(f"Loaded {len(df):,} ECG samples.")

# --------------------------------------------------
# VALIDATE DATA
# --------------------------------------------------
y_test = df["label"].astype(str).str.strip().to_numpy()
X_df = df.drop(columns=["label"])

try:
    X = X_df.apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float32
    )
except Exception as e:
    st.error(f"ECG columns must contain numeric values: {e}")
    st.stop()

if X.shape[1] != 180:
    st.error(
        f"Expected 180 ECG values per sample; found {X.shape[1]} columns."
    )
    st.stop()

if not np.isfinite(X).all():
    st.error("The ECG dataset contains missing or infinite values.")
    st.stop()

unknown_labels = sorted(set(y_test) - set(classes))
if unknown_labels:
    st.error(
        f"Labels not recognized by the trained model: {unknown_labels}. "
        f"Expected labels from: {classes}"
    )
    st.stop()

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
# IMPORTANT: This per-segment normalization must match
# the preprocessing used during CNN training.
means = X.mean(axis=1, keepdims=True)
stds = X.std(axis=1, keepdims=True)
X = (X - means) / np.maximum(stds, 1e-6)

X_cnn = X[..., np.newaxis]

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------
with st.spinner("Running CNN predictions..."):
    probabilities = model.predict(
        X_cnn, batch_size=256, verbose=0
    )

if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
    st.error(
        "The model output shape does not match the number of classes."
    )
    st.stop()

y_pred = np.array([
    classes[i] for i in np.argmax(probabilities, axis=1)
])

# --------------------------------------------------
# SAMPLE SELECTION
# --------------------------------------------------
st.sidebar.header("ECG Sample Selection")

sample_number = st.sidebar.number_input(
    "Select ECG Sample Index",
    min_value=1,
    max_value=len(X),
    value=1,
    step=1,
)

idx = int(sample_number) - 1
st.sidebar.write(f"Total test samples: {len(X):,}")

# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

try:
    y_binary = label_binarize(y_test, classes=classes)
    if len(classes) == 2 and y_binary.shape[1] == 1:
        y_binary = np.column_stack([1 - y_binary[:, 0], y_binary[:, 0]])

    auc = roc_auc_score(
        y_binary,
        probabilities,
        multi_class="ovr",
        average="weighted",
    )
except Exception:
    auc = np.nan

correct = int(np.sum(y_test == y_pred))
incorrect = len(y_test) - correct

st.header("5-Class Classification Performance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
c2.metric("Weighted F1", f"{weighted_f1:.4f}")
c3.metric("Macro F1", f"{macro_f1:.4f}")
c4.metric(
    "AUC-ROC",
    f"{auc:.4f}" if np.isfinite(auc) else "N/A",
)

st.write(f"**Correct predictions:** {correct:,}")
st.write(f"**Incorrect predictions:** {incorrect:,}")

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    labels=classes,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).T
report_df = report_df.reindex(classes)
report_df = report_df.rename(columns={
    "precision": "Precision",
    "recall": "Recall",
    "f1-score": "F1-Score",
    "support": "Support",
})

st.dataframe(
    report_df[["Precision", "Recall", "F1-Score", "Support"]].style.format({
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}",
        "Support": "{:.0f}",
    }),
    use_container_width=True,
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=classes)

fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(cm)
ax.set_xticks(range(len(classes)), classes)
ax.set_yticks(range(len(classes)), classes)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("CNN Confusion Matrix")

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# SELECTED ECG SAMPLE
# --------------------------------------------------
st.divider()
st.header(f"ECG Sample {sample_number:,}")

predicted = y_pred[idx]
actual = y_test[idx]

a, b, c = st.columns(3)
a.metric("Predicted Class", predicted)
b.metric("Actual Class", actual)
c.metric(
    "Result",
    "Correct" if predicted == actual else "Incorrect",
)

left, right = st.columns(2)

with left:
    st.subheader("ECG Waveform")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(X[idx])
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("CNN Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Class": classes,
        "Probability": probabilities[idx],
    })
    st.bar_chart(prob_df.set_index("Class"))

# --------------------------------------------------
# GRADIENT-BASED EXPLANATION
# --------------------------------------------------
st.subheader("Explainable AI Analysis")

try:
    x_one = tf.convert_to_tensor(X_cnn[idx:idx + 1])

    with tf.GradientTape() as tape:
        tape.watch(x_one)
        scores = model(x_one, training=False)
        selected_score = scores[:, np.argmax(probabilities[idx])]

    gradient = tape.gradient(selected_score, x_one)

    if gradient is None:
        raise ValueError("The model gradient could not be calculated.")

    saliency = np.abs(gradient.numpy()[0, :, 0])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(saliency)
    ax.set_xlabel("ECG Sample Index")
    ax.set_ylabel("Absolute Gradient")
    ax.set_title("CNN Gradient-Based Saliency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Gradient-based saliency highlights input locations that influence "
        "the selected CNN score. It does not establish clinical causality."
    )
except Exception as e:
    st.error(f"Could not calculate the explanation: {e}")

# --------------------------------------------------
# DOWNLOAD PREDICTIONS
# --------------------------------------------------
st.subheader("Download Predictions")

results = pd.DataFrame({
    "sample_index": np.arange(1, len(X) + 1),
    "actual_class": y_test,
    "predicted_class": y_pred,
    "correct": y_test == y_pred,
})

for i, cls in enumerate(classes):
    results[f"probability_{cls}"] = probabilities[:, i]

st.download_button(
    "Download All Predictions as CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="cnn_ecg_predictions.csv",
    mime="text/csv",
)
