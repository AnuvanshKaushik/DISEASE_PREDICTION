from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


# -------------------------------
# Paths (edit if your files differ)
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
DRIVE_DATA_URL = "https://drive.google.com/uc?export=download&id=1zZDM-2IKRNmewMt1rY58m86WpvbD1EYk"
DESC_PATH = BASE_DIR / "data" / "Description.csv"
MODEL_PATH = BASE_DIR / "models" / "optimized_disease_prediction_model.h5"


# -------------------------------
# Cached loaders
# -------------------------------
@st.cache_data(show_spinner=False)
def load_dataset() -> Tuple[pd.DataFrame, List[str], LabelEncoder]:
    try:
        df = pd.read_csv(DRIVE_DATA_URL)
    except Exception as e:
        raise RuntimeError(f"Failed loading dataset from Google Drive: {e}")

    if "diseases" not in df.columns:
        raise ValueError("Dataset must contain a 'diseases' column.")

    symptom_cols = df.columns.drop("diseases")
    all_symptoms = symptom_cols.tolist()

    label_encoder = LabelEncoder()
    label_encoder.fit(df["diseases"])

    return df, all_symptoms, label_encoder


@st.cache_data(show_spinner=False)
def load_description() -> pd.DataFrame:
    if not DESC_PATH.exists():
        raise FileNotFoundError(f"Description file missing at {DESC_PATH}")
    df = pd.read_csv(DESC_PATH)
    df.columns = (
        df.columns.str.strip()
        .str.replace(" / ", "_")
        .str.replace(" ", "_")
        .str.lower()
    )
    return df


@st.cache_resource(show_spinner=False)
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# -------------------------------
# Prediction helpers
# -------------------------------
def symptoms_to_vector(user_symptoms: List[str], all_symptoms: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Map user-entered symptoms to the dataset columns.
    Returns vector, matched labels, and unmatched inputs.
    """
    num_symptoms = len(all_symptoms)
    vector = np.zeros(num_symptoms, dtype=float)
    matched = []
    missing = []

    for raw in user_symptoms:
        symptom = raw.strip().lower()
        if not symptom:
            continue
        hits = [idx for idx, col in enumerate(all_symptoms) if symptom in col.lower()]
        if hits:
            for idx in hits:
                vector[idx] = 1
                matched.append(all_symptoms[idx])
        else:
            missing.append(raw)
    return vector.reshape(1, -1), matched, missing


def predict_disease(model: tf.keras.Model, vector: np.ndarray, encoder: LabelEncoder, top_k: int = 3):
    probs = model.predict(vector, verbose=0)[0]
    top_indices = probs.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append(
            {
                "label": encoder.inverse_transform([idx])[0],
                "prob": float(probs[idx]),
            }
        )
    return results


def lookup_details(desc_df: pd.DataFrame, disease: str) -> pd.Series | None:
    if "disease_name" not in desc_df.columns:
        return None
    match = desc_df[
        desc_df["disease_name"].str.lower().str.contains(disease.lower(), na=False)
    ]
    return match.iloc[0] if not match.empty else None


# -------------------------------
# UI sections
# -------------------------------
def render_header():
    st.set_page_config(
        page_title="AI Disease Predictor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def add_styles():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --bg: #0c1220;
            --panel: #11192b;
            --accent: #6ac8ff;
            --accent-2: #925bff;
            --text: #e8f0ff;
            --muted: #9fb0c9;
        }
        .stApp, .main {
            background: radial-gradient(circle at 10% 20%, rgba(146,91,255,0.12), transparent 25%),
                        radial-gradient(circle at 80% 0%, rgba(106,200,255,0.12), transparent 25%),
                        var(--bg) !important;
            color: var(--text) !important;
        }
        .stMarkdown, .stText, .stMetric, h1, h2, h3, h4, h5, h6, p, label, span {
            color: var(--text) !important;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #0c1220 100%);
            border-right: 1px solid rgba(255,255,255,0.05);
            color: #fff;
        }
        section[data-testid="stSidebar"] * {
            color: #fff !important;
        }
        /* Hero */
        .hero {
            padding: 24px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(146,91,255,0.85), rgba(106,200,255,0.85));
            color: white;
            box-shadow: 0 18px 50px rgba(0,0,0,0.4);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .hero:hover {
            transform: translateY(-2px);
            box-shadow: 0 24px 60px rgba(0,0,0,0.45);
        }
        /* Cards */
        .card, .stContainer {
            padding: 18px;
            border-radius: 14px;
            background: var(--panel);
            border: 1px solid rgba(255,255,255,0.05);
            box-shadow: 0 12px 30px rgba(0,0,0,0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .card:hover, .stContainer:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 42px rgba(0,0,0,0.32);
        }
        .glass {
            background: linear-gradient(135deg, rgba(146,91,255,0.18), rgba(106,200,255,0.18));
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 18px 44px rgba(0,0,0,0.35);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .glass:hover {
            transform: translateY(-2px);
            box-shadow: 0 22px 56px rgba(0,0,0,0.4);
        }
        .metric {
            font-size: 32px;
            font-weight: 700;
            color: var(--accent);
        }
        .badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(146,91,255,0.18);
            color: var(--text);
            margin: 4px 6px 0 0;
            font-size: 12px;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 999px;
            background: linear-gradient(120deg, var(--accent), var(--accent-2));
            color: #fff;
            font-weight: 600;
            box-shadow: 0 12px 28px rgba(0,0,0,0.35);
        }
        .small-title {
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 1px;
            color: var(--muted);
        }
        /* Buttons */
        button[kind="primary"], .stButton>button {
            background: linear-gradient(120deg, var(--accent), var(--accent-2));
            color: #fff;
            border: none;
            box-shadow: 0 10px 24px rgba(0,0,0,0.3);
            transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(0,0,0,0.35);
            opacity: 0.95;
        }
        button[kind="secondary"] {
            background: var(--panel);
            color: var(--accent);
            border: 1px solid var(--accent);
        }
        /* Progress */
        .stProgress .st-bo { background: rgba(255,255,255,0.08) !important; }
        .stProgress .st-bq { background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important; }
        /* Cards layout */
        .detail-card {
            padding: 16px;
            border-radius: 14px;
            background: var(--panel);
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 12px 30px rgba(0,0,0,0.3);
        }
        /* Hide default block gaps to tighten layout */
        .block-container {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(num_symptoms: int, num_diseases: int):
    st.markdown(
        f"""
        <div class="hero">
            <div class="small-title">AI Health Assistant</div>
            <h1 style="margin: 0 0 8px 0;">Disease & Recommendation Predictor</h1>
            <p style="margin: 0 0 16px 0; font-size: 16px; max-width: 720px;">
                Enter symptoms to get instant AI predictions, confidence scores, and doctor recommendations — powered by your trained TensorFlow model.
            </p>
            <div style="display:flex;gap:16px;flex-wrap:wrap;">
                <div>
                    <div class="small-title">Symptoms Catalogued</div>
                    <div class="metric">{num_symptoms}</div>
                </div>
                <div>
                    <div class="small-title">Diseases Covered</div>
                    <div class="metric">{num_diseases}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(all_symptoms: List[str]) -> List[str]:
    st.sidebar.header("🩺 Symptoms")
    st.sidebar.write("Start typing to filter. You can also paste a comma-separated list below.")

    chosen = st.sidebar.multiselect(
        "Pick symptoms",
        options=all_symptoms,
        default=None,
        help="Choose as many as you like",
    )

    free_text = st.sidebar.text_area(
        "Or paste symptoms (comma separated)",
        value="",
        placeholder="fever, headache, cough",
    )
    if free_text:
        chosen.extend([s.strip() for s in free_text.split(",") if s.strip()])

    st.sidebar.info(
        "Tip: The model expects symptoms from the training dataset. "
        "We'll do fuzzy matching on partial names."
    )
    return chosen


def render_results(results, matched, missing, desc_df):
    if not matched:
        st.warning("No valid symptoms matched the dataset. Please refine your input.")
        return

    st.success(f"Matched symptoms: {', '.join(sorted(set(matched)))}")
    if missing:
        st.caption(f"Not recognized: {', '.join(missing)}")

    top = results[0]
    st.subheader("Top Prediction")
    st.markdown(
        f"""
        <div class="glass" style="padding:18px;border-radius:16px;margin-bottom:16px;">
            <div class="pill">Top match · {top['prob']*100:.1f}%</div>
            <h2 style="margin:10px 0 0 0;">{top["label"]}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Top 3 candidates")
    cols = st.columns(len(results))
    for col, item in zip(cols, results):
        with col:
            st.markdown(f"**{item['label']}**")
            st.progress(item["prob"])
            st.caption(f"{item['prob']*100:.1f}%")

    details = lookup_details(desc_df, top["label"])
    st.divider()
    st.subheader("Disease Details")
    if details is not None:
        with st.container(border=True):
            st.markdown(f"**Disease Name:** {details.get('disease_name', 'N/A')}")
            st.markdown(
                f"**Short Description:** {details.get('short_description', 'N/A')}"
            )
            st.markdown(
                f"**General Precautions:** {details.get('general_precautions', 'N/A')}"
            )
            st.markdown(
                f"**Recommended Doctor:** {details.get('recommended_doctor_specialist', 'N/A')}"
            )
    else:
        st.info("No additional details found in Description.csv.")


def main():
    render_header()
    add_styles()
    try:
        _, all_symptoms, label_encoder = load_dataset()
        desc_df = load_description()
        model = load_model()
    except Exception as exc:  # pragma: no cover - user-facing errors
        st.error(f"Setup issue: {exc}")
        st.stop()

    num_symptoms = len(all_symptoms)
    num_diseases = len(label_encoder.classes_)

    render_hero(num_symptoms, num_diseases)
    chosen_symptoms = render_sidebar(all_symptoms)

    st.divider()
    tabs = st.tabs(["Predict", "About the Model", "How it Works", "FAQ"])

    with tabs[0]:
        st.header("Prediction")
        st.write("Click predict after selecting at least one symptom.")

        preset_col1, preset_col2 = st.columns(2)
        with preset_col1:
            if st.button("Use common cold symptoms 🧣", use_container_width=True):
                chosen_symptoms.extend(
                    [s for s in ["fever", "cough", "runny nose", "fatigue"] if s not in chosen_symptoms]
                )
        with preset_col2:
            if st.button("Use stomach upset symptoms 🍽️", use_container_width=True):
                chosen_symptoms.extend(
                    [s for s in ["nausea", "vomiting", "abdominal pain"] if s not in chosen_symptoms]
                )

        if st.button("Predict", type="primary", use_container_width=True):
            vector, matched, missing = symptoms_to_vector(chosen_symptoms, all_symptoms)
            if not matched:
                st.warning("Please provide at least one valid symptom from the list.")
                st.stop()

            results = predict_disease(model, vector, label_encoder)
            render_results(results, matched, missing, desc_df)
        else:
            st.info("Awaiting your symptoms. Use the sidebar to select or type them.")

    with tabs[1]:
        st.header("About the Model")
        st.markdown(
            """
            - TensorFlow dense network with L2 regularization and dropout for stability.  
            - Trained on your augmented symptoms/disease dataset; standardized inputs, label-encoded outputs.  
            - Predicts top-3 diseases with confidence scores and links to description metadata.
            """
        )
        st.markdown(
            f"""
            **Coverage:** {num_symptoms} symptoms across {num_diseases} diseases.  
            **Model file:** `{MODEL_PATH.name}`  
            **Data sources:** `{DATA_PATH.name}`, `{DESC_PATH.name}`
            """
        )

    with tabs[2]:
        st.header("How it Works")
        st.markdown(
            """
            1. **Input** — You select or type symptoms; we fuzzy-match to known symptom columns.  
            2. **Vectorize** — Matched symptoms become a binary vector aligned with training columns.  
            3. **Predict** — The trained model outputs class probabilities; we show the top 3.  
            4. **Explain** — The top disease is paired with description, precautions, and doctor guidance.  
            """
        )

    with tabs[3]:
        st.header("FAQ")
        st.markdown(
            """
            - **Why do some symptoms show as not recognized?**  
              Only symptoms present in the training dataset can be matched; try partial words.  
            - **Can I upload new data?**  
              Replace the CSVs in `data/` and keep column names consistent.  
            - **Model not loading?**  
              Ensure `models/optimized_disease_prediction_model.h5` exists and matches the training schema.  
            - **Is this a medical diagnosis?**  
              No. Always consult licensed medical professionals for decisions.
            """
        )


if __name__ == "__main__":
    main()

