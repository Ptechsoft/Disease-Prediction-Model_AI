import os
import re
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Nigeria Health Assistant", layout="centered")

MODEL_PATH = "disease_predictor.pkl"
TREATMENT_CSV = "treatment_plans_updated.csv"
DRUG_CSV = "drug_suggestions_updated.csv"

# Default OpenRouter model (free). Can be overridden in secrets.toml
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.2-3b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# =========================================================
# HELPERS
# =========================================================
def norm_disease(x: str) -> str:
    return str(x).strip().lower()


def is_symptom_command(text: str) -> bool:
    """Prediction triggers ONLY with: symptoms: ..."""
    return bool(re.match(r"^\s*symptoms\s*:\s*", text.strip().lower()))


def extract_symptom_payload(text: str) -> str:
    """Everything after 'symptoms:'"""
    return re.sub(r"^\s*symptoms\s*:\s*", "", text.strip(), flags=re.IGNORECASE)


def parse_symptom_text(text: str):
    """
    "fever, headache, joint pain" -> ["fever","headache","joint_pain"]
    """
    parts = [p.strip().lower().replace(" ", "_") for p in text.split(",")]
    return [p for p in parts if p]


def build_input_row(symptoms_list, feature_list):
    """Binary row: features in training order."""
    row = {f: 0 for f in feature_list}
    for s in symptoms_list:
        if s in row:
            row[s] = 1
    return pd.DataFrame([row], columns=feature_list)


# =========================================================
# LOAD MODEL + CSVs
# =========================================================
@st.cache_resource
def load_model_bundle(model_path: str):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    # Avoid feature mismatch: use model feature names when possible
    if hasattr(model, "get_booster"):  # XGBoost
        feature_list = model.get_booster().feature_names
    elif hasattr(model, "feature_names_in_"):
        feature_list = list(model.feature_names_in_)
    else:
        feature_list = bundle.get("features", None)

    if feature_list is None:
        raise ValueError("Could not determine feature list. Re-save model with features.")

    return model, label_encoder, feature_list


@st.cache_data
def load_plans_and_drugs(treatment_csv: str, drug_csv: str):
    treatment_df = pd.read_csv(treatment_csv)
    drug_df = pd.read_csv(drug_csv)

    treatment_map = {
        norm_disease(row["disease"]): row.dropna().to_dict()
        for _, row in treatment_df.iterrows()
    }
    drug_map = {
        norm_disease(row["disease"]): row.dropna().to_dict()
        for _, row in drug_df.iterrows()
    }
    return treatment_map, drug_map


def predict(symptoms_list, model, label_encoder, feature_list, treatment_map, drug_map):
    X = build_input_row(symptoms_list, feature_list)

    pred_class = int(model.predict(X)[0])
    disease = label_encoder.inverse_transform([pred_class])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))

    key = norm_disease(disease)
    plan = treatment_map.get(key, {"note": "No treatment plan found in dataset. Consult a clinician."})
    drugs = drug_map.get(key, {"note": "No drug suggestions found in dataset. Consult a clinician/pharmacist."})

    return disease, confidence, plan, drugs


# =========================================================
# OPENROUTER CLIENT (SAFE KEY LOADING)
# =========================================================
def get_openrouter_client_and_model():
    """
    Tries:
      1) environment variable OPENROUTER_API_KEY
      2) st.secrets["OPENROUTER_API_KEY"] (Streamlit secrets.toml)
    If key not found, returns (None, model_name) without crashing.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["OPENROUTER_API_KEY"]
        except Exception:
            api_key = None

    model_name = os.getenv("OPENROUTER_MODEL")

    if not model_name:
        try:
            model_name = st.secrets.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        except Exception:
            model_name = DEFAULT_OPENROUTER_MODEL

    if not api_key:
        return None, model_name

    # OpenRouter is OpenAI-compatible; set base_url
    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL
    )
    return client, model_name


def gpt_reply_openrouter(client: OpenAI, model_name: str, messages: list) -> str:
    """
    OpenRouter works well with Chat Completions format.
    """
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.4
    )
    return resp.choices[0].message.content


# =========================================================
# SESSION STATE
# =========================================================
if "stage" not in st.session_state:
    st.session_state.stage = "details"
if "chat" not in st.session_state:
    st.session_state.chat = []  # chat messages: role/content
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None
if "symptoms_submitted" not in st.session_state:
    st.session_state.symptoms_submitted = False


# =========================================================
# LOAD APP RESOURCES
# =========================================================
model, label_encoder, feature_list = load_model_bundle(MODEL_PATH)
treatment_map, drug_map = load_plans_and_drugs(TREATMENT_CSV, DRUG_CSV)
trained_diseases = list(label_encoder.classes_)

client, router_model = get_openrouter_client_and_model()


# =========================================================
# UI HEADER
# =========================================================
st.title("🇳🇬 Nigeria Health Assistant (OpenRouter + ML Decision Support) BUILT BY NWADIGO PRAISE AKACHUKWU")
st.info(
    "⚠️ This is a supportive screening tool and does NOT replace a medical professional. "
    "For diagnosis and prescriptions, consult a clinician."
)

# =========================================================
# STAGE 1: PERSONAL DETAILS
# =========================================================
if st.session_state.stage == "details":
    st.subheader("Step 1: Personal Details")

    with st.form("details_form"):
        name = st.text_input("Full Name")
        gender = st.selectbox("Gender", ["Select...", "Male", "Female", "Other", "Prefer not to say"])
        mode = st.radio("Provide", ["Age", "Date of Birth"], horizontal=True)

        if mode == "Age":
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
        else:
            dob = st.date_input("Date of Birth", value=date(2000, 1, 1))
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        proceed = st.form_submit_button("Continue to Chat")

    if proceed:
        if not name.strip():
            st.error("Please enter your name.")
            st.stop()
        if gender == "Select...":
            st.error("Please select a gender.")
            st.stop()

        st.session_state.user_profile = {"name": name.strip(), "gender": gender, "age": int(age)}
        st.session_state.stage = "chat"
        st.session_state.chat = []
        st.session_state.symptoms_submitted = False

        system_prompt = f"""
You are a helpful health assistant for Nigerian users.

You can answer general questions and explain health topics.
You MUST NOT predict diseases outside the trained ML model.

STRICT RULES:
1) Only run disease prediction if the user types: "symptoms: ..." (comma-separated).
2) When predicting, ONLY present the ML model output (no invented conditions).
3) If asked about diseases outside the trained list, explain limitation and list supported diseases.
4) Always include a short disclaimer that this does not replace a clinician.

SUPPORTED DISEASES:
{", ".join(trained_diseases)}

To get a prediction, instruct the user to type:
symptoms: fever, headache, fatigue
"""

        st.session_state.chat.append({"role": "system", "content": system_prompt})
        st.session_state.chat.append({
            "role": "assistant",
            "content": (
                f"Hello {st.session_state.user_profile['name']} 👋\n\n"
                "You can ask me any health question.\n\n"
                "✅ For ML prediction, type your symptoms like:\n"
                "**symptoms: fever, headache, fatigue**\n\n"
                f"🧠 This ML model is trained on: **{', '.join(trained_diseases)}**."
            )
        })
        st.rerun()

# =========================================================
# STAGE 2: CHAT
# =========================================================
else:
    st.subheader("Chat")

    # Show chat history (exclude system from display)
    for msg in st.session_state.chat:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if client is None:
        st.warning(
            "OpenRouter chat is currently OFF because no OPENROUTER_API_KEY was found.\n\n"
            "✅ You can still get ML predictions by typing:\n"
            "**symptoms: fever, headache, fatigue**\n\n"
            "To enable chat, set OPENROUTER_API_KEY in .streamlit/secrets.toml or as an environment variable."
        )

    user_text = st.chat_input("Ask anything… or type: symptoms: fever, headache, ...")

    if user_text:
        st.session_state.chat.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        # -----------------------------
        # ML PREDICTION PATH
        # -----------------------------
        if is_symptom_command(user_text):
            if st.session_state.symptoms_submitted:
                assistant_msg = (
                    "✅ Symptoms have already been submitted in this session.\n\n"
                    "If you want to run another prediction, refresh the page to start a new session.\n\n"
                    "⚠️ This is not a replacement for a clinician."
                )
                st.session_state.chat.append({"role": "assistant", "content": assistant_msg})
                with st.chat_message("assistant"):
                    st.write(assistant_msg)
                st.stop()

            symptom_payload = extract_symptom_payload(user_text)
            symptoms_list = parse_symptom_text(symptom_payload)

            unknown = [s for s in symptoms_list if s not in feature_list]
            known = [s for s in symptoms_list if s in feature_list]

            if len(known) == 0:
                assistant_msg = (
                    "I couldn't match your symptoms to the model’s symptom list.\n\n"
                    "Example symptoms the model understands:\n"
                    f"`{', '.join(feature_list[:10])}`\n\n"
                    "Try:\n"
                    "**symptoms: fever, headache, fatigue**\n\n"
                    "⚠️ This is not a replacement for a clinician."
                )
                st.session_state.chat.append({"role": "assistant", "content": assistant_msg})
                with st.chat_message("assistant"):
                    st.write(assistant_msg)
                st.stop()

            disease, confidence, plan, drugs = predict(
                known, model, label_encoder, feature_list, treatment_map, drug_map
            )

            conf_text = "N/A" if confidence is None else f"{confidence:.2%}"
            plan_lines = "\n".join([f"- **{k}**: {v}" for k, v in plan.items()])
            drug_lines = "\n".join([f"- **{k}**: {v}" for k, v in drugs.items()])

            unknown_text = ""
            if unknown:
                unknown_text = (
                    "\n\n⚠️ **Unrecognized symptoms ignored:** "
                    + ", ".join(unknown)
                    + "\n(These symptoms are not in the model’s symptom list yet.)"
                )

            assistant_msg = (
                f"✅ **Prediction Result (ML Model)**\n\n"
                f"**Predicted Disease:** {disease}\n"
                f"**Confidence:** {conf_text}\n\n"
                f"### Treatment Plan\n{plan_lines}\n\n"
                f"### Drug Suggestions\n{drug_lines}"
                f"{unknown_text}\n\n"
                "⚠️ This is a supportive tool. Please consult a licensed medical professional for confirmation and prescriptions."
            )

            st.session_state.chat.append({"role": "assistant", "content": assistant_msg})
            st.session_state.symptoms_submitted = True

            with st.chat_message("assistant"):
                st.write(assistant_msg)
            st.stop()

        # -----------------------------
        # OPENROUTER GENERAL CHAT PATH
        # -----------------------------
        if client is None:
            assistant_msg = (
                "OpenRouter chat is not active because OPENROUTER_API_KEY is not set.\n\n"
                "✅ You can still get predictions with:\n"
                "**symptoms: fever, headache, fatigue**\n\n"
                "⚠️ This is not a replacement for a clinician."
            )
        else:
            # Keep chat history reasonable: system + last 20 messages
            system_msgs = [m for m in st.session_state.chat if m["role"] == "system"]
            recent_msgs = [m for m in st.session_state.chat if m["role"] != "system"][-20:]
            messages = system_msgs + recent_msgs

            assistant_msg = gpt_reply_openrouter(client, router_model, messages)

            # Ensure disclaimer appears
            if "not a replacement" not in assistant_msg.lower() and "does not replace" not in assistant_msg.lower():
                assistant_msg += "\n\n⚠️ This is not a replacement for a clinician."

        st.session_state.chat.append({"role": "assistant", "content": assistant_msg})
        with st.chat_message("assistant"):
            st.write(assistant_msg)