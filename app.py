
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


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ECG Arrhythmia Detection | CNN",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ ECG Arrhythmia Detection Using CNN")
st.subheader("Explainable AI Framework for ECG-Based Arrhythmia Diagnosis")

st.warning(
    "Academic research prototype only. "
    "Predictions are not intended for clinical diagnosis."
)


# =========================================================
# 2. LOAD MODEL, ENCODER AND METADATA
# =========================================================

@st.cache_resource
def load_assets():
    model_path = "ecg_cnn_model.keras"
    encoder_path = "ecg_label_encoder.pkl"
    metadata_path = "model_metadata.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model file: {model_path}"
        )

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Missing encoder file: {encoder_path}"
        )

    model = tf.keras.models.load_model(model_path)

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    metadata = {}

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, encoder, metadata


try:
    model, label_encoder, metadata = load_assets()
except Exception as exc:
    st.error("Unable to load the model files.")
    st.exception(exc)
    st.stop()


# =========================================================
# 3. EXTRACT CLASS NAMES SAFELY
# =========================================================

def extract_class_names(encoder, metadata):
    # Standard sklearn LabelEncoder
    if hasattr(encoder, "classes_"):
        return [str(v) for v in encoder.classes_]

    # Dictionary-based encoder
    if isinstance(encoder, dict):
        for key in ("classes", "class_names", "classes_"):
            if key in encoder:
                values = encoder[key]

                if isinstance(values, dict):
                    # Handle mappings such as {"N": 0, "S": 1}
                    try:
                        values = [
                            key for key, value in
                            sorted(
                                values.items(),
                                key=lambda item: int(item[1])
                            )
                        ]
                    except (TypeError, ValueError):
                        values = list(values.keys())

                if values is not None:
                    return [str(v) for v in values]

        # Some encoders are stored as label-to-index mappings.
        if encoder and all(
            isinstance(v, (int, np.integer))
            for v in encoder.values()
        ):
            try:
                return [
                    str(k)
                    for k, v in sorted(
                        encoder.items(),
                        key=lambda item: int(item[1])
                    )
                ]
            except (TypeError, ValueError):
                pass

    # Metadata class list
    for key in ("class_names", "classes", "labels"):
        values = metadata.get(key)

        if isinstance(values, list) and values:
            return [str(v) for v in values]

    # Fallback: only use a known, explicitly stored class mapping.
    # Do not silently assume a class order.
    return None


class_names = extract_class_names(label_encoder, metadata)

if not class_names:
    st.error(
        "Could not determine the class order from the saved "
        "encoder or metadata. Please check how the encoder "
        "was created in the training notebook."
    )
    st.write("Encoder object type:", str(type(label_encoder)))
    st.write("Metadata keys:", list(metadata.keys()))
    st.stop()


num_classes = len(class_names)

if len(model.output_shape) < 2:
    st.error(f"Unexpected model output shape: {model.output_shape}")
    st.stop()

if int(model.output_shape[-1]) != num_classes:
    st.error(
        f"The CNN has {model.output_shape[-1]} output classes, "
        f"but the encoder provides {num_classes} class names. "
        "The saved encoder and model may not belong together."
    )
    st.stop()

st.success("CNN model, encoder and class names loaded.")

with st.expander("Model information"):
    st.write("Model input shape:", model.input_shape)
    st.write("Model output shape:", model.output_shape)
    st.write("Class order:", class_names)
    st.write("Metadata:", metadata)


# =========================================================
# 4. LOAD TEST CSV PARTS
# =========================================================

@st.cache_data
def load_test_data():
    folder = "ecg_test_parts"

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            "Folder 'ecg_test_parts' not found. "
            "Place this folder beside app.py in your repository."
        )

    paths = glob.glob(os.path.join(folder, "*.csv"))

    def natural_key(path):
        name = os.path.basename(path)
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)
        ]

    paths = sorted(paths, key=natural_key)

    if not paths:
        raise FileNotFoundError(
            "No CSV files were found inside ecg_test_parts."
        )

    frames = []

    for path in paths:
        part = pd.read_csv(path)

        # Remove accidental pandas index columns.
        part = part.loc[
            :,
            ~part.columns.astype(str).str.match(r"^Unnamed:")
        ]

        if "label" not in part.columns:
            st.warning(
                f"Skipping {os.path.basename(path)}: "
                "no 'label' column."
            )
            continue

        frames.append(part)

    if not frames:
        raise ValueError(
            "None of the CSV files contains a 'label' column."
        )

    df = pd.concat(frames, ignore_index=True)

    # ECG feature columns are expected to be named 0, 1, ... 179.
    feature_cols = [
        c for c in df.columns
        if c != "label"
        and str(c).strip().isdigit()
    ]

    feature_cols = sorted(feature_cols, key=lambda c: int(str(c)))

    if not feature_cols:
        raise ValueError(
            "Could not identify ECG feature columns. "
            "Expected signal columns named 0 through 179."
        )

    for column in feature_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["label"] = df["label"].astype(str).str.strip()

    df = df.dropna(
        subset=feature_cols + ["label"]
    ).reset_index(drop=True)

    return df, feature_cols, len(paths)


try:
    df, feature_cols, file_count = load_test_data()
except Exception as exc:
    st.error("Unable to load the ECG test dataset.")
    st.exception(exc)
    st.stop()


st.info(
    f"Loaded {len(df):,} valid ECG samples from "
    f"{file_count} CSV files."
)

if len(feature_cols) != 180:
    st.error(
        f"Expected 180 features per ECG sample; "
        f"found {len(feature_cols)}. Check the CSV files."
    )
    st.stop()

if df.empty:
    st.error("The test dataset contains no valid samples.")
    st.stop()


# =========================================================
# 5. MAP TRUE LABELS TO THE MODEL'S CLASS ORDER
# =========================================================

raw_labels = df["label"].to_numpy()


def encode_labels(labels, encoder, class_names):
    # Use the encoder's own transform method where possible.
    if hasattr(encoder, "transform"):
        try:
            return np.asarray(
                encoder.transform(labels.astype(str)),
                dtype=int
            )
        except Exception:
            pass

    # Labels already contain class names.
    class_to_id = {
        str(name): index
        for index, name in enumerate(class_names)
    }

    if all(str(label) in class_to_id for label in labels):
        return np.array(
            [class_to_id[str(label)] for label in labels],
            dtype=int
        )

    # Labels may be integer IDs stored in the CSV.
    try:
        numeric = pd.to_numeric(
            pd.Series(labels), errors="raise"
        ).to_numpy()

        if np.all(numeric == numeric.astype(int)):
            numeric = numeric.astype(int)

            if (
                np.all(numeric >= 0)
                and np.all(numeric < len(class_names))
            ):
                return numeric
    except (ValueError, TypeError):
        pass

    raise ValueError(
        "CSV labels do not match the saved encoder's classes "
        "or valid integer class IDs. Verify the training label "
        "mapping before evaluating the model."
    )


try:
    y_true = encode_labels(
        raw_labels,
        label_encoder,
        class_names
    )
except Exception as exc:
    st.error("Unable to encode the true labels.")
    st.exception(exc)
    st.stop()


# =========================================================
# 6. PREPROCESS ECG SIGNALS
# =========================================================

X_raw = df[feature_cols].to_numpy(dtype=np.float32)

if not np.isfinite(X_raw).all():
    st.error("ECG signals contain NaN or infinite values.")
    st.stop()

# IMPORTANT:
# This row-wise normalization must match your CNN training
# notebook. Change it if training used another preprocessing
# method, such as a saved scaler or no normalization.

row_means = X_raw.mean(axis=1, keepdims=True)
row_stds = X_raw.std(axis=1, keepdims=True)
row_stds[row_stds < 1e-8] = 1.0

X_normalized = (X_raw - row_means) / row_stds

if len(model.input_shape) == 3:
    X_model = X_normalized[..., np.newaxis]
elif len(model.input_shape) == 2:
    X_model = X_normalized
else:
    st.error(
        f"Unsupported model input shape: {model.input_shape}"
    )
    st.stop()


# =========================================================
# 7. CNN PREDICTIONS
# =========================================================

with st.spinner("Predicting ECG classes..."):
    try:
        probabilities = model.predict(
            X_model,
            batch_size=256,
            verbose=0
        )
    except Exception as exc:
        st.error("CNN prediction failed.")
        st.exception(exc)
        st.stop()

if probabilities.ndim != 2:
    st.error(
        f"Unexpected prediction array shape: {probabilities.shape}"
    )
    st.stop()

if probabilities.shape[1] != num_classes:
    st.error(
        "The number of predicted classes does not match "
        "the encoder's class count."
    )
    st.stop()

y_pred = np.argmax(probabilities, axis=1)

true_names = np.array(class_names)[y_true]
predicted_names = np.array(class_names)[y_pred]


# =========================================================
# 8. METRICS
# =========================================================

accuracy = accuracy_score(y_true, y_pred)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Test Samples", f"{len(df):,}")
c2.metric("Accuracy", f"{accuracy:.2%}")
c3.metric("Weighted F1", f"{weighted_f1:.4f}")
c4.metric("Macro F1", f"{macro_f1:.4f}")


# =========================================================
# 9. CONFUSION MATRIX
# =========================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes))
)

fig_cm, ax_cm = plt.subplots(figsize=(9, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax_cm,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax_cm.set_title("CNN Arrhythmia Classification")
ax_cm.set_xlabel("Predicted class")
ax_cm.set_ylabel("Actual class")

plt.setp(
    ax_cm.get_xticklabels(),
    rotation=45,
    ha="right"
)

fig_cm.tight_layout()
st.pyplot(fig_cm)
plt.close(fig_cm)


# =========================================================
# 10. CLASSIFICATION REPORT
# =========================================================

st.subheader("Classification Report")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# =========================================================
# 11. ROC-AUC
# =========================================================

st.subheader("ROC-AUC Score")

try:
    if num_classes == 2:
        auc = roc_auc_score(
            y_true,
            probabilities[:, 1]
        )
    else:
        y_true_binary = label_binarize(
            y_true,
            classes=list(range(num_classes))
        )

        auc = roc_auc_score(
            y_true_binary,
            probabilities,
            average="weighted",
            multi_class="ovr"
        )

    st.metric("Weighted ROC-AUC", f"{auc:.4f}")

except Exception as exc:
    st.info(
        "ROC-AUC could not be calculated for this dataset. "
        f"Details: {exc}"
    )


# =========================================================
# 12. ECG WAVEFORM VIEWER
# =========================================================

st.subheader("ECG Waveform Viewer")

sample_idx = st.slider(
    "Select ECG sample",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    key="ecg_sample_slider"
)

signal = X_raw[sample_idx]

fig_wave, ax_wave = plt.subplots(figsize=(12, 4))

ax_wave.plot(
    np.arange(len(signal)),
    signal,
    linewidth=1.5
)

ax_wave.set_title(
    f"Sample {sample_idx} | "
    f"Actual: {true_names[sample_idx]} | "
    f"Predicted: {predicted_names[sample_idx]}"
)

ax_wave.set_xlabel("Sample index")
ax_wave.set_ylabel("Amplitude")
ax_wave.grid(True, alpha=0.3)

fig_wave.tight_layout()
st.pyplot(fig_wave)
plt.close(fig_wave)


# =========================================================
# 13. CLASS PROBABILITIES
# =========================================================

st.subheader("Prediction Probabilities")

sample_probs = probabilities[sample_idx]

prob_df = pd.DataFrame({
    "Class": class_names,
    "Probability": sample_probs
}).sort_values(
    "Probability",
    ascending=False
)

fig_prob, ax_prob = plt.subplots(figsize=(8, 4))

ax_prob.bar(
    prob_df["Class"],
    prob_df["Probability"]
)

ax_prob.set_ylim(0, 1)
ax_prob.set_xlabel("Class")
ax_prob.set_ylabel("Probability")
ax_prob.set_title(
    f"Class probabilities for sample {sample_idx}"
)
ax_prob.grid(axis="y", alpha=0.3)

fig_prob.tight_layout()
st.pyplot(fig_prob)
plt.close(fig_prob)

st.dataframe(
    prob_df.reset_index(drop=True).round(4),
    use_container_width=True
)


# =========================================================
# 14. GRADIENT-BASED SALIENCY
# =========================================================

st.subheader("Gradient-Based Signal Saliency")

st.caption(
    "This is a basic gradient explanation, not SHAP. "
    "It highlights input positions that influence the "
    "selected model output locally."
)

try:
    input_tensor = tf.convert_to_tensor(
        X_model[sample_idx:sample_idx + 1],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        output = model(input_tensor, training=False)
        predicted_class = tf.argmax(
            output[0],
            output_type=tf.int32
        )
        score = output[0, predicted_class]

    gradients = tape.gradient(score, input_tensor)

    if gradients is None:
        st.info("Gradients are unavailable for this model.")
    else:
        saliency = np.abs(
            gradients.numpy().squeeze()
        )

        if (
            saliency.ndim == 1
            and len(saliency) == len(signal)
        ):
            fig_sal, ax_sal = plt.subplots(figsize=(12, 4))

            ax_sal.plot(
                signal,
                label="ECG signal",
                linewidth=1.2
            )

            ax_sal.set_xlabel("Sample index")
            ax_sal.set_ylabel("Amplitude")
            ax_sal.set_title(
                "ECG Signal and Gradient Saliency"
            )

            ax2 = ax_sal.twinx()

            ax2.plot(
                saliency,
                linestyle="--",
                label="Absolute gradient"
            )

            ax2.set_ylabel("Gradient magnitude")

            fig_sal.tight_layout()
            st.pyplot(fig_sal)
            plt.close(fig_sal)
        else:
            st.info(
                "Saliency output does not match the ECG signal shape."
            )

except Exception as exc:
    st.info(f"Could not generate saliency: {exc}")


# =========================================================
# 15. DOWNLOAD PREDICTIONS
# =========================================================

st.subheader("Download Predictions")

results_df = pd.DataFrame({
    "actual_label": true_names,
    "predicted_label": predicted_names,
    "correct": y_true == y_pred
})

for index, class_name in enumerate(class_names):
    results_df[f"probability_{class_name}"] = (
        probabilities[:, index]
    )

st.download_button(
    label="Download prediction results as CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="ecg_cnn_predictions.csv",
    mime="text/csv"
)


# =========================================================
# 16. DATASET INFORMATION
# =========================================================

with st.expander("Dataset information"):
    st.write("Number of CSV files:", file_count)
    st.write("Valid ECG samples:", len(df))
    st.write("Features per ECG:", len(feature_cols))
    st.write("Class order:", class_names)

    st.write("True-label distribution:")

    counts = pd.Series(
        true_names
    ).value_counts().reindex(
        class_names,
        fill_value=0
    )

    st.dataframe(
        counts.rename("Sample count"),
        use_container_width=True
    )

st.caption(
    "Research prototype. Verify the training preprocessing, "
    "label mapping, and test split before interpreting metrics."
)
