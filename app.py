
import os
import glob
import pickle
import json
import re

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ECG Arrhythmia Detection | CNN",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ ECG Arrhythmia Detection Using CNN")
st.caption(
    "Explainable AI Framework for ECG-Based Arrhythmia Diagnosis"
)

st.warning(
    "Academic research prototype only. "
    "Predictions are not intended for clinical diagnosis."
)


# --------------------------------------------------
# 2. LOAD MODEL AND LABEL ENCODER
# --------------------------------------------------

@st.cache_resource
def load_assets():
    model_path = "ecg_cnn_model.keras"
    encoder_path = "ecg_label_encoder.pkl"
    metadata_path = "model_metadata.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Label encoder not found: {encoder_path}"
        )

    model = tf.keras.models.load_model(model_path)

    with open(encoder_path, "rb") as file:
        encoder = pickle.load(file)

    metadata = {}

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as file:
            metadata = json.load(file)

    return model, encoder, metadata


try:
    model, label_encoder, metadata = load_assets()
except Exception as exc:
    st.error(f"Could not load the model assets: {exc}")
    st.stop()


# The encoder's class order must match the order used
# when the CNN was trained.
class_names = [
    str(value) for value in label_encoder.classes_
]

num_classes = len(class_names)

st.success("CNN model and label encoder loaded successfully.")

with st.expander("Model information"):
    st.write("Model input shape:", model.input_shape)
    st.write("Model output shape:", model.output_shape)
    st.write("Encoder class order:", class_names)
    st.write("Metadata:", metadata)


# --------------------------------------------------
# 3. LOAD ALL TEST CSV FILES
# --------------------------------------------------

@st.cache_data
def load_test_data():
    folder = "ecg_test_parts"

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            "Folder 'ecg_test_parts' was not found. "
            "Create it in the same repository as app.py."
        )

    files = sorted(
        glob.glob(os.path.join(folder, "*.csv")),
        key=lambda path: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", os.path.basename(path))
        ],
    )

    if not files:
        raise FileNotFoundError(
            "No CSV files found inside ecg_test_parts."
        )

    frames = []

    for path in files:
        part = pd.read_csv(path)

        # Remove an index column accidentally saved by pandas.
        part = part.loc[
            :,
            ~part.columns.astype(str).str.match(
                r"^Unnamed:"
            ),
        ]

        if "label" not in part.columns:
            st.warning(
                f"Skipping {os.path.basename(path)}: "
                "the required 'label' column is missing."
            )
            continue

        frames.append(part)

    if not frames:
        raise ValueError(
            "No valid CSV files containing a 'label' column "
            "were found."
        )

    df = pd.concat(frames, ignore_index=True)

    # Identify numeric signal columns and sort them in
    # their original order: 0, 1, 2 ... 179.
    feature_cols = [
        column
        for column in df.columns
        if column != "label"
        and str(column).strip().lstrip("-").isdigit()
    ]

    feature_cols = sorted(
        feature_cols,
        key=lambda column: int(str(column)),
    )

    if not feature_cols:
        # Fallback for CSVs whose signal columns have names
        # other than numeric indices.
        feature_cols = [
            column for column in df.columns
            if column != "label"
        ]

    for column in feature_cols:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df["label"] = df["label"].astype(str).str.strip()

    df = df.dropna(
        subset=feature_cols + ["label"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid ECG samples remain after cleaning."
        )

    return df, feature_cols, len(files)


try:
    df, feature_cols, file_count = load_test_data()
except Exception as exc:
    st.error(f"Could not load the test dataset: {exc}")
    st.stop()


st.info(
    f"Loaded {len(df):,} valid ECG samples "
    f"from {file_count} CSV file(s)."
)

if len(feature_cols) != 180:
    st.error(
        f"Expected 180 ECG signal features, but found "
        f"{len(feature_cols)}. Check your CSV column format."
    )
    st.stop()

if len(class_names) != int(model.output_shape[-1]):
    st.error(
        "The number of encoder classes does not match "
        "the CNN model's output classes."
    )
    st.stop()


# --------------------------------------------------
# 4. PREPARE SIGNALS AND TRUE LABELS
# --------------------------------------------------

X_raw = df[feature_cols].to_numpy(dtype=np.float32)
raw_labels = df["label"].to_numpy()

# Convert true labels to the same class IDs used during
# model training.
#
# This supports CSV labels that contain either the original
# class names or numeric IDs produced by LabelEncoder.

try:
    if set(raw_labels).issubset(set(class_names)):
        y_true = label_encoder.transform(raw_labels)
    else:
        numeric_labels = pd.to_numeric(
            pd.Series(raw_labels), errors="raise"
        ).to_numpy()

        if not np.all(numeric_labels == numeric_labels.astype(int)):
            raise ValueError("Labels are not integer class IDs.")

        y_true = numeric_labels.astype(int)

        if (
            np.any(y_true < 0)
            or np.any(y_true >= num_classes)
        ):
            raise ValueError(
                "Numeric labels are outside the encoder's "
                "valid class range."
            )

except Exception as exc:
    st.error(
        "Could not map the CSV labels to the model's class IDs. "
        "Make sure the test CSV uses the same label encoding "
        "as the CNN training notebook."
    )
    st.exception(exc)
    st.stop()


# Row-wise z-score normalization.
# IMPORTANT: this must match the preprocessing used during
# CNN training. If training used a different method, replace
# this section with the exact training preprocessing.

means = X_raw.mean(axis=1, keepdims=True)
stds = X_raw.std(axis=1, keepdims=True)
stds[stds < 1e-8] = 1.0

X_normalized = (X_raw - means) / stds

# Typical CNN input shape: (samples, 180, 1).
# Do not reshape this way if your trained model expects a
# different input shape.
if len(model.input_shape) == 3:
    X_model = X_normalized[..., np.newaxis]
elif len(model.input_shape) == 2:
    X_model = X_normalized
else:
    st.error(
        f"Unsupported CNN input shape: {model.input_shape}"
    )
    st.stop()


# --------------------------------------------------
# 5. PREDICTIONS
# --------------------------------------------------

with st.spinner("Running CNN predictions..."):
    try:
        probabilities = model.predict(
            X_model, batch_size=256, verbose=0
        )
    except Exception as exc:
        st.error(f"CNN prediction failed: {exc}")
        st.stop()

if probabilities.ndim != 2:
    st.error(
        "Expected model.predict() to return a "
        "2D array of class probabilities."
    )
    st.stop()

y_pred = np.argmax(probabilities, axis=1)

if probabilities.shape[1] != num_classes:
    st.error(
        "Prediction output classes do not match "
        "the saved label encoder."
    )
    st.stop()

# Human-readable labels in the encoder's original class order.
true_names = np.array(class_names)[y_true]
predicted_names = np.array(class_names)[y_pred]


# --------------------------------------------------
# 6. PERFORMANCE METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Test Samples", f"{len(df):,}")
col2.metric("Accuracy", f"{accuracy:.2%}")
col3.metric("Weighted F1", f"{weighted_f1:.4f}")
col4.metric("Macro F1", f"{macro_f1:.4f}")


# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------

st.subheader("Confusion Matrix")

all_class_ids = list(range(num_classes))

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=all_class_ids,
)

fig_cm, ax_cm = plt.subplots(figsize=(9, 7))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(
    ax=ax_cm,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax_cm.set_title("CNN Arrhythmia Classification")
ax_cm.set_xlabel("Predicted class")
ax_cm.set_ylabel("Actual class")

plt.setp(
    ax_cm.get_xticklabels(),
    rotation=45,
    ha="right",
)

fig_cm.tight_layout()
st.pyplot(fig_cm)
plt.close(fig_cm)


# --------------------------------------------------
# 8. CLASSIFICATION REPORT
# --------------------------------------------------

st.subheader("Classification Report")

report_dict = classification_report(
    y_true,
    y_pred,
    labels=all_class_ids,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report_dict).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True,
)


# --------------------------------------------------
# 9. ROC-AUC (WHEN CALCULABLE)
# --------------------------------------------------

st.subheader("ROC-AUC Score")

try:
    if num_classes == 2:
        auc = roc_auc_score(
            y_true, probabilities[:, 1]
        )
    else:
        y_true_binary = label_binarize(
            y_true, classes=all_class_ids
        )

        auc = roc_auc_score(
            y_true_binary,
            probabilities,
            average="weighted",
            multi_class="ovr",
        )

    st.metric("Weighted ROC-AUC", f"{auc:.4f}")

except Exception as exc:
    st.info(
        "ROC-AUC could not be calculated for this dataset. "
        "One or more classes may be missing or the labels "
        f"may not match the model outputs. Details: {exc}"
    )


# --------------------------------------------------
# 10. ECG WAVEFORM
# --------------------------------------------------

st.subheader("ECG Waveform Viewer")

sample_idx = st.slider(
    "Select an ECG sample",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    key="waveform_sample",
)

signal = X_raw[sample_idx]

fig_wave, ax_wave = plt.subplots(figsize=(12, 4))

ax_wave.plot(
    np.arange(len(signal)),
    signal,
    linewidth=1.5,
)

ax_wave.set_title(
    f"ECG Sample {sample_idx} | "
    f"Actual: {true_names[sample_idx]} | "
    f"Predicted: {predicted_names[sample_idx]}"
)

ax_wave.set_xlabel("Sample index")
ax_wave.set_ylabel("Signal amplitude")
ax_wave.grid(True, alpha=0.3)

fig_wave.tight_layout()
st.pyplot(fig_wave)
plt.close(fig_wave)


# --------------------------------------------------
# 11. CLASS PROBABILITIES
# --------------------------------------------------

st.subheader("Prediction Probabilities")

sample_probs = probabilities[sample_idx]

prob_df = pd.DataFrame({
    "Class": class_names,
    "Probability": sample_probs,
})

prob_df = prob_df.sort_values(
    "Probability", ascending=False
)

fig_prob, ax_prob = plt.subplots(figsize=(8, 4))

ax_prob.bar(
    prob_df["Class"],
    prob_df["Probability"],
)

ax_prob.set_ylim(0, 1)
ax_prob.set_ylabel("Probability")
ax_prob.set_xlabel("Arrhythmia class")
ax_prob.set_title(
    f"Class probabilities for sample {sample_idx}"
)
ax_prob.grid(axis="y", alpha=0.3)

fig_prob.tight_layout()
st.pyplot(fig_prob)
plt.close(fig_prob)

st.dataframe(
    prob_df.reset_index(drop=True).round(4),
    use_container_width=True,
)


# --------------------------------------------------
# 12. GRADIENT-BASED SALIENCY
# --------------------------------------------------

st.subheader("Gradient-Based Signal Saliency")

st.caption(
    "This visualization highlights signal positions with "
    "larger input gradients for the selected prediction. "
    "It is a basic explanation method, not SHAP."
)

try:
    input_tensor = tf.convert_to_tensor(
        X_model[sample_idx:sample_idx + 1],
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        output = model(input_tensor, training=False)
        target_class = tf.argmax(
            output[0], output_type=tf.int32
        )
        target_score = output[0, target_class]

    gradients = tape.gradient(
        target_score, input_tensor
    )

    if gradients is None:
        st.info(
            "Gradients were unavailable for this model."
        )
    else:
        saliency = np.abs(
            gradients.numpy().squeeze()
        )

        if saliency.ndim == 1 and len(saliency) == len(signal):
            fig_sal, ax_sal = plt.subplots(figsize=(12, 4))

            ax_sal.plot(
                signal,
                label="ECG signal",
                linewidth=1.2,
            )

            ax_sal.set_xlabel("Sample index")
            ax_sal.set_ylabel("Amplitude")
            ax_sal.set_title(
                "ECG Signal with Gradient Saliency"
            )

            ax_sal2 = ax_sal.twinx()
            ax_sal2.plot(
                saliency,
                linestyle="--",
                label="Absolute gradient",
            )
            ax_sal2.set_ylabel("Gradient magnitude")

            fig_sal.tight_layout()
            st.pyplot(fig_sal)
            plt.close(fig_sal)
        else:
            st.info(
                "The gradient shape did not match the ECG signal."
            )

except Exception as exc:
    st.info(f"Saliency could not be generated: {exc}")


# --------------------------------------------------
# 13. DOWNLOAD PREDICTIONS
# --------------------------------------------------

st.subheader("Download Predictions")

results_df = pd.DataFrame({
    "actual_label": true_names,
    "predicted_label": predicted_names,
    "correct": y_true == y_pred,
})

for index, class_name in enumerate(class_names):
    results_df[f"probability_{class_name}"] = (
        probabilities[:, index]
    )

csv_bytes = results_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download prediction results as CSV",
    data=csv_bytes,
    file_name="ecg_cnn_predictions.csv",
    mime="text/csv",
)


# --------------------------------------------------
# 14. DATASET INFORMATION
# --------------------------------------------------

with st.expander("Dataset information"):
    st.write("Total valid samples:", len(df))
    st.write("Number of signal features:", len(feature_cols))
    st.write("CSV files loaded:", file_count)
    st.write("Classes:", class_names)

    st.write("True-label distribution:")

    label_counts = pd.Series(
        true_names
    ).value_counts().reindex(
        class_names, fill_value=0
    )

    st.dataframe(
        label_counts.rename("Sample count"),
        use_container_width=True,
    )

st.caption(
    "Research prototype • Validate the test split, label encoding, "
    "and normalization against the training notebook before "
    "reporting model performance."
)
