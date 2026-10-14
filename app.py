import json, pickle
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st, tensorflow as tf
import matplotlib.pyplot as plt

HERE=Path(__file__).parent
st.set_page_config(page_title="ECG CNN Arrhythmia Classifier",page_icon="💓",layout="wide")
st.title("Explainable AI Framework for ECG-Based Arrhythmia Diagnosis")
st.caption("1D CNN · MIT-BIH heartbeat classes · Educational research prototype")

@st.cache_resource
def load_assets():
    model=tf.keras.models.load_model(HERE/"ecg_cnn_model.keras")
    with open(HERE/"ecg_label_encoder.pkl","rb") as f: enc=pickle.load(f)
    with open(HERE/"model_metadata.json") as f: meta=json.load(f)
    return model,enc["class_names"],meta

try:
    model,CLASSES,META=load_assets()
except Exception as e:
    st.error(f"Model files missing or could not load: {e}")
    st.stop()

L=int(META.get("segment_length",180))
descriptions={"N":"Normal beat group","S":"Supraventricular ectopic beat group",
              "V":"Ventricular ectopic beat group","F":"Fusion beat group","Q":"Unknown or paced beat group"}
st.warning("For education/research only. Not a medical device; do not use for diagnosis or treatment.")
mode=st.radio("Input",["Synthetic demo (not a real ECG)","Upload CSV with one 180-sample segment"])
segment=None
if mode.startswith("Synthetic"):
    t=np.linspace(-1,1,L)
    segment=(0.12*np.sin(2*np.pi*2*t)+0.7*np.exp(-((t+0.22)/0.07)**2)
             -0.25*np.exp(-((t-0.02)/0.025)**2)+1.1*np.exp(-((t-0.05)/0.035)**2)
             +0.35*np.exp(-((t-0.32)/0.12)**2)).astype("float32")
    st.info("Synthetic demo only; this is not a real patient's ECG.")
else:
    f=st.file_uploader("CSV file (180 numeric values total)",type=["csv"])
    if f:
        try:
            df=pd.read_csv(f,header=None)
            vals=pd.to_numeric(df.stack(),errors="coerce").dropna().to_numpy(dtype="float32")
            if len(vals)!=L: st.error(f"Expected exactly {L} numeric values, got {len(vals)}.")
            else: segment=vals
        except Exception as e: st.error(f"Could not read CSV: {e}")

if segment is not None:
    sd=float(segment.std()); normalized=np.zeros_like(segment) if sd<1e-6 else (segment-float(segment.mean()))/sd
    p=model.predict(normalized.reshape(1,L,1),verbose=0)[0]; idx=int(np.argmax(p)); label=CLASSES[idx]
    a,b=st.columns(2)
    with a:
        st.subheader("ECG segment")
        fig,ax=plt.subplots(figsize=(8,3)); ax.plot(segment); ax.set_xlabel("Sample index"); ax.set_ylabel("Amplitude"); ax.grid(alpha=.25)
        st.pyplot(fig); plt.close(fig)
    with b:
        st.subheader("Prediction")
        st.metric("Predicted class",label); st.write(descriptions.get(label,"Heartbeat class"))
        st.bar_chart(pd.DataFrame({"Probability":p},index=CLASSES))
        st.dataframe(pd.DataFrame({"Class":CLASSES,"Probability":[round(float(x),4) for x in p]}),hide_index=True)
    st.subheader("CNN input saliency")
    st.write("Gradient-based sensitivity view: high values indicate input samples to which the predicted score is locally sensitive. This is not SHAP and is not evidence of clinical causality.")
    xt=tf.convert_to_tensor(normalized.reshape(1,L,1))
    with tf.GradientTape() as tape:
        tape.watch(xt); output=model(xt,training=False); score=output[:,idx]
    grad=tape.gradient(score,xt).numpy()[0,:,0]
    fig,ax=plt.subplots(figsize=(10,3)); ax.plot(np.abs(grad)); ax.set_xlabel("Sample index"); ax.set_ylabel("Absolute gradient"); ax.grid(alpha=.25)
    st.pyplot(fig); plt.close(fig)
