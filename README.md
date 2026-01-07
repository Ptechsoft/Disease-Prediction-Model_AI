# DISEASE-PREDICTION-MODEL + AI(OPENAI)

Project Overview

Nigeria Health Assistant is an AI-driven health screening and decision-support application designed to help users and healthcare professionals quickly assess possible illnesses based on symptoms, access treatment guidance, and receive drug suggestions before meeting a medical professional.

The system combines:

Machine Learning (ML) for disease prediction

Large Language Models (LLMs) for conversational reasoning and health education

Structured medical datasets for treatment and drug recommendations

⚠️ This tool does NOT replace medical professionals.
It is intended to provide supportive guidance only.

How the System Works
1️⃣ User Information Collection

The app first collects:

- Full name

- Gender

- Age or Date of Birth

- This helps personalize the interaction and ensures responsible usage.

2️⃣ Conversational Chat Interface (AI-Powered)

- Users interact through a chat-based interface where they can:

- Ask general health questions

- Learn about symptoms and diseases

- Get guidance on when to see a clinician

- This chat system is powered by a GPT-style language model via OpenRouter AI.

3️⃣ Disease Prediction (ML-Only)

Disease prediction is triggered only when the user enters symptoms in this format:

`symptoms: fever, headache, fatigue`

**Key rules:**

- Symptoms must be comma-separated

- Prediction happens once per session

- The model only predicts diseases it was trained on

4️⃣ Treatment Plan & Drug Suggestions

After prediction:

A treatment plan is retrieved from a structured CSV dataset

Drug suggestions are provided based on standard clinical guidance

If a disease is not in the dataset, the app clearly informs the user

Diseases Supported (Current Version)

The ML model is trained on a limited but clinically relevant set of diseases, including:

`Malaria`, `Typhoid`, `Cholera`, `Tuberculosis`, `COVID-19`, `Diabetes`, `Hypertension`, `Asthma` `Lassa Fever` `Sickle Cell Disease`
**More diseases will be added in future versions.**

**Technologies Used**
**Machine Learning:**
- Random Forest
- XGBoost
- CatBoost
- Scikit-learn
- AI / NLP
- OpenRouter AI (LLM access)
- Meta LLaMA 3.2 3B Instruct (free tier)
**Backend & Deployment:**
- Python
- Streamlit
- Joblib
**Data Handling:**
- Pandas
- NumPy

**Disclaimer**
**This application provides supportive health information only.**
**It is not a diagnostic tool and does not replace licensed healthcare professionals.**
**If you experience severe or persistent symptoms, please seek immediate medical care.**# Disease-Prediction-Model

Link to test the app : https://disease-prediction-modelai-app.streamlit.app/
