
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
    page_title="ECG Arrhythmia Detection",
    page_icon="❤️",
    layout="wide",
)

st.title("❤️ ECG Arrhythmia Detection Using CNN")
st.subheader(
    "Explainable AI Framework for ECG-Based Arrhythmia Diagnosis"
)

st.warning(
    "Academic research prototype only. "
    "This application is not intended for clinical diagnosis."
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
            f"Model file not found: {model_path}"
        )

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Encoder file not found: {encoder_path}"
        )

    model = tf.keras.models.load_model(model_path)

    with open(encoder_path, "rb") as file:
        encoder = pickle.load(file)

    metadata = {}

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    return model, encoder, metadata


try:
    model, label_encoder, metadata = load_assets()
except Exception as exc:
    st.error("Unable to load the model files.")
    st.exception(exc)
    st.stop()


# =========================================================
# 3. DETERMINE CLASS NAMES
# =========================================================

def extract_class_names(encoder, metadata):
    # Standard sklearn LabelEncoder
    if hasattr(encoder, "classes_"):
        return [str(value) for value in encoder.classes_]

    # Dictionary-based encoder
    if isinstance(encoder, dict):
        for key in ("classes", "class_names", "classes_"):
            if key not in encoder:
                continue

            values = encoder[key]

            if isinstance(values, dict):
                try:
                    values = [
                        key for key, value in sorted(
                            values.items(),
                            key=lambda item: int(item[1])
                        )
                    ]
                except (TypeError, ValueError):
                    values = list(values.keys())

            if isinstance(values, (list, tuple, np.ndarray)):
                return [str(value) for value in values]

        # Handle mappings such as {"N": 0, "S": 1}.
        if encoder and all(
            isinstance(value, (int, np.integer))
            for value in encoder.values()
        ):
            try:
                return [
                    str(key)
                    for key, value in sorted(
                        encoder.items(),
                        key=lambda item: int(item[1])
                    )
                ]
            except (TypeError, ValueError):
                pass

    # Metadata class names
    for key in ("class_names", "classes", "labels"):
        values = metadata.get(key)

        if isinstance(values, list) and values:
            return [str(value) for value in values]

    return None


class_names = extract_class_names(
    label_encoder,
    metadata
)

if not class_names:
    st.error(
        "Cannot determine the class order from the saved encoder "
        "or metadata. Check how the encoder was created in the "
        "training notebook."
    )
    st.write("Encoder type:", str(type(label_encoder)))
    st.write("Metadata keys:", list(metadata.keys()))
    st.stop()


num_classes = len(class_names)

if len(model.output_shape) < 2:
    st.error(
        f"Unexpected model output shape: {model.output_shape}"
    )
    st.stop()

if int(model.output_shape[-1]) != num_classes:
    st.error(
        f"Model output classes: {model.output_shape[-1]}. "
        f"Encoder classes: {num_classes}. "
        "The model and encoder may not match."
    )
    st.stop()

st.success("CNN model and class information loaded.")

with st.expander("Model information"):
    st.write("Model input shape:", model.input_shape)
    st.write("Model output shape:", model.output_shape)
    st.write("Class order:", class_names)
    st.write("Metadata:", metadata)


# =========================================================
# 4. LOAD ECG CSV FILES
# =========================================================

@st.cache_data
def load_test_data():
    folder = "ecg_test_parts"

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Folder '{folder}' not found in the repository."
        )

    paths = glob.glob(
        os.path.join(folder, "*.csv")
    )

    if not paths:
        raise FileNotFoundError(
            f"No CSV files found inside '{folder}'."
        )

    def natural_key(path):
        name = os.path.basename(path)
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)
        ]

    paths = sorted(paths, key=natural_key)

    label_candidates = {
        "label",
        "labels",
        "target",
        "class",
        "category",
        "beat_type",
        "y",
    }

    standardized_features = [
        f"feature_{i}" for i in range(180)
    ]

    frames = []

    for path in paths:
        part = pd.read_csv(path)

        # Remove accidental pandas index columns.
        part = part.loc[
            :,
            ~part.columns.astype(str).str.match(
                r"^Unnamed:\s*\d+$",
                case=False
            )
        ]

        # Ignore surrounding spaces and capitalization.
        normalized_names = {
            str(column).strip().lower(): column
            for column in part.columns
        }

        label_col = next(
            (
                normalized_names[name]
                for name in label_candidates
                if name in normalized_names
            ),
            None
        )

        if label_col is None:
            raise ValueError(
                f"Could not identify the label column in "
                f"{os.path.basename(path)}. "
                f"Columns found: {list(part.columns)[:30]}"
            )

        # Recognize numeric columns and common feature names.
        feature_cols = []

        for column in part.columns:
            if column == label_col:
                continue

            name = str(column).strip().lower()

            if re.fullmatch(r"\d+(?:\.0+)?", name):
                feature_cols.append(column)

            elif re.fullmatch(
                r"(?:sample|signal|feature|x)[_\s-]*\d+",
                name
            ):
                feature_cols.append(column)

        # If names are unfamiliar, accept the remaining columns
        # only when exactly 180 remain.
        other_cols = [
            column for column in part.columns
            if column != label_col
        ]

        if len(feature_cols) != 180:
            if len(other_cols) == 180:
                feature_cols = other_cols
            else:
                raise ValueError(
                    f"File: {os.path.basename(path)}. "
                    f"Found {len(feature_cols)} recognizable signal "
                    f"columns and {len(other_cols)} non-label columns. "
                    f"Expected exactly 180 signal features. "
                    f"Columns: {list(part.columns)[:30]}"
                )

        def feature_number(column):
            name = str(column).strip().lower()

            match = re.search(
                r"(\d+(?:\.0+)?)$",
                name
            )

            if match:
                return float(match.group(1))

            return float("inf")

        # Order signal columns by their numeric suffix.
        feature_cols = sorted(
            feature_cols,
            key=feature_number
        )

        rename_map = {
            column: f"feature_{index}"
            for index, column in enumerate(feature_cols)
        }

        rename_map[label_col] = "label"

        part = part.rename(columns=rename_map)

        for column in standardized_features:
            part[column] = pd.to_numeric(
                part[column],
                errors="coerce"
            )

        part["label"] = (
            part["label"]
            .astype(str)
            .str.strip()
        )

        part = part.dropna(
            subset=standardized_features + ["label"]
        )

        frames.append(
            part[standardized_features + ["label"]]
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid ECG samples remain after cleaning."
        )

    return df, standardized_features, len(paths)


try:
    df, feature_cols, file_count = load_test_data()
except Exception as exc:
    st.error("Unable to load the ECG test dataset.")
    st.exception(exc)
    st.stop()


st.info(
    f"Loaded {len(df):,} ECG samples from "
    f"{file_count} CSV files."
)

if len(feature_cols) != 180:
    st.error(
        f"Expected 180 features per ECG sample, "
        f"but found {len(feature_cols)}."
    )
    st.stop()


# =========================================================
# 5. ENCODE ACTUAL LABELS
# =========================================================

def encode_labels(labels, encoder, class_names):
    labels = np.asarray(labels).astype(str)

    # Prefer the saved encoder's own mapping.
    if hasattr(encoder, "transform"):
        try:
            return np.asarray(
                encoder.transform(labels),
                dtype=int
            )
        except Exception:
            pass

    class_to_id = {
        str(name): index
        for index, name in enumerate(class_names)
    }

    # Labels already contain class names.
    if all(label in class_to_id for label in labels):
        return np.array(
            [class_to_id[label] for label in labels],
            dtype=int
        )

    # Labels may already be encoded integer IDs.
    try:
        numeric = pd.to_numeric(
            pd.Series(labels),
            errors="raise"
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
        "CSV labels do not match the saved encoder classes "
        "or valid integer IDs. Verify the label mapping "
        "used during training."
    )


try:
    y_true = encode_labels(
        df["label"].to_numpy(),
        label_encoder,
        class_names
    )
except Exception as exc:
    st.error("Unable to encode the ECG labels.")
    st.exception(exc)
    st.stop()


# =========================================================
# 6. PREPROCESS ECG SIGNALS
# =========================================================

X_raw = df[feature_cols].to_numpy(
    dtype=np.float32
)

if not np.isfinite(X_raw).all():
    st.error(
        "The ECG data contains NaN or infinite values."
    )
    st.stop()

# IMPORTANT:
# This per-sample normalization must match the training code.
# Replace it if the model was trained with a different method.

row_means = X_raw.mean(
    axis=1,
    keepdims=True
)

row_stds = X_raw.std(
    axis=1,
    keepdims=True
)

row_stds[row_stds < 1e-8] = 1.0

X_normalized = (
    X_raw - row_means
) / row_stds

if len(model.input_shape) == 3:
    if model.input_shape[1] not in (None, 180):
        st.error(
            f"Model expects {model.input_shape[1]} time steps, "
            "but the dataset has 180."
        )
        st.stop()

    X_model = X_normalized[..., np.newaxis]

elif len(model.input_shape) == 2:
    X_model = X_normalized

else:
    st.error(
        f"Unsupported model input shape: {model.input_shape}"
    )
    st.stop()


# =========================================================
# 7. GENERATE CNN PREDICTIONS
# =========================================================

with st.spinner("Running CNN predictions..."):
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
        f"Unexpected prediction shape: {probabilities.shape}"
    )
    st.stop()

if probabilities.shape[1] != num_classes:
    st.error(
        "The model's prediction count does not match "
        "the encoder's class count."
    )
    st.stop()

y_pred = np.argmax(
    probabilities,
    axis=1
)

true_names = np.asarray(class_names)[y_true]
predicted_names = np.asarray(class_names)[y_pred]


# =========================================================
# 8. OVERALL PERFORMANCE METRICS
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

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

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Test Samples",
    f"{len(df):,}"
)

col2.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col3.metric(
    "Weighted F1",
    f"{weighted_f1:.4f}"
)

col4.metric(
    "Macro F1",
    f"{macro_f1:.4f}"
)


# =========================================================
# 9. CONFUSION MATRIX
# =========================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes))
)

fig_cm, ax_cm = plt.subplots(
    figsize=(9, 7)
)

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

ax_cm.set_title(
    "CNN Arrhythmia Classification"
)

ax_cm.set_xlabel(
    "Predicted class"
)

ax_cm.set_ylabel(
    "Actual class"
)

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

report_df = pd.DataFrame(
    report
).transpose()

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

    st.metric(
        "Weighted ROC-AUC",
        f"{auc:.4f}"
    )

except Exception as exc:
    st.info(
        "ROC-AUC could not be calculated. "
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

fig_wave, ax_wave = plt.subplots(
    figsize=(12, 4)
)

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

ax_wave.set_xlabel(
    "Sample index"
)

ax_wave.set_ylabel(
    "Amplitude"
)

ax_wave.grid(
    True,
    alpha=0.3
)

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

fig_prob, ax_prob = plt.subplots(
    figsize=(8, 4)
)

ax_prob.bar(
    prob_df["Class"],
    prob_df["Probability"]
)

ax_prob.set_ylim(0, 1)

ax_prob.set_xlabel("Class")
ax_prob.set_ylabel("Probability")

ax_prob.set_title(
    f"Prediction probabilities for sample {sample_idx}"
)

ax_prob.grid(
    axis="y",
    alpha=0.3
)

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
    "It indicates how sensitive the selected output is "
    "to small changes in the input signal."
)

try:
    input_tensor = tf.convert_to_tensor(
        X_model[sample_idx:sample_idx + 1],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)

        output = model(
            input_tensor,
            training=False
        )

        predicted_class = tf.argmax(
            output[0],
            output_type=tf.int32
        )

        score = output[0, predicted_class]

    gradients = tape.gradient(
        score,
        input_tensor
    )

    if gradients is None:
        st.info(
            "Gradients are unavailable for this model."
        )

    else:
        saliency = np.abs(
            gradients.numpy().squeeze()
        )

        if (
            saliency.ndim == 1
            and len(saliency) == len(signal)
        ):
            fig_sal, ax_sal = plt.subplots(
                figsize=(12, 4)
            )

            ax_sal.plot(
                signal,
                label="ECG signal",
                linewidth=1.2
            )

            ax_sal.set_xlabel(
                "Sample index"
            )

            ax_sal.set_ylabel(
                "Amplitude"
            )

            ax_sal.set_title(
                "ECG Signal and Gradient Saliency"
            )

            ax2 = ax_sal.twinx()

            ax2.plot(
                saliency,
                linestyle="--",
                label="Absolute gradient"
            )

            ax2.set_ylabel(
                "Gradient magnitude"
            )

            fig_sal.tight_layout()

            st.pyplot(fig_sal)

            plt.close(fig_sal)

        else:
            st.info(
                "Saliency dimensions do not match "
                "the ECG waveform."
            )

except Exception as exc:
    st.info(
        f"Could not generate saliency: {exc}"
    )


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
    results_df[
        f"probability_{class_name}"
    ] = probabilities[:, index]

st.download_button(
    label="Download prediction results as CSV",
    data=results_df.to_csv(
        index=False
    ).encode("utf-8"),
    file_name="ecg_cnn_predictions.csv",
    mime="text/csv"
)


# =========================================================
# 16. DATASET INFORMATION
# =========================================================

with st.expander("Dataset Information"):
    st.write(
        "Number of CSV files:",
        file_count
    )

    st.write(
        "Valid ECG samples:",
        len(df)
    )

    st.write(
        "Features per sample:",
        len(feature_cols)
    )

    st.write(
        "Class order:",
        class_names
    )

    st.write("True label distribution:")

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


# =========================================================
# 17. FOOTER
# =========================================================

st.divider()

st.caption(
    "ECG Arrhythmia Detection | CNN Research Prototype. "
    "Validate the training preprocessing, class mapping, "
    "and independent test split before reporting results. "
    "Not for clinical use."
)
