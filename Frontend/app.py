from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

import os

# 1. PROJECT SETTINGS

ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = ROOT / "data" / "prediction_history.csv"

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


HISTORY_COLUMNS = [
    "timestamp",
    "candidate_role",
    "job_title",
    "candidate_experience_years",
    "job_required_experience",
    "experience_gap",
    "match_probability",
    "prediction",
    "result",
]


# 2. PAGE CONFIGURATION


st.set_page_config(
    page_title="Smart Recruitment & Talent Analytics",
    page_icon="👥",
    layout="wide",
)

st.title("AI-Powered Smart Recruitment")
st.caption("Candidate Matching & Talent Analytics System")


# 3. PREDICTION HISTORY HELPERS

def load_history():
    """Load saved prediction history from CSV."""

    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        history = pd.read_csv(HISTORY_PATH)

        # Handle an empty or older history file safely.
        for column in HISTORY_COLUMNS:
            if column not in history.columns:
                history[column] = None

        return history[HISTORY_COLUMNS]

    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def save_prediction(record):
    """Append one successful prediction to the CSV file."""

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = load_history()
    new_record = pd.DataFrame([record], columns=HISTORY_COLUMNS)

    updated_history = pd.concat(
        [history, new_record],
        ignore_index=True,
    )

    updated_history.to_csv(HISTORY_PATH, index=False)


# 4. SIDEBAR NAVIGATION

page = st.sidebar.radio(
    "Navigation",
    [
        "Candidate Matching",
        "Talent Analytics Dashboard",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Powered by LSTM, FastAPI and Streamlit")


# 5. CANDIDATE MATCHING PAGE

if page == "Candidate Matching":

    st.header("Candidate Information")

    with st.form("recruitment_form"):

        col1, col2 = st.columns(2)

        with col1:
            candidate_role = st.text_input(
                "Candidate Role",
                placeholder="e.g. Data Analyst",
            )

            candidate_education = st.text_input(
                "Candidate Education",
                placeholder="e.g. BSc Mathematics",
            )

            candidate_experience = st.number_input(
                "Candidate Experience (years)",
                min_value=0.0,
                value=3.0,
                step=1.0,
            )

        with col2:
            candidate_skills = st.text_area(
                "Candidate Skills",
                placeholder="e.g. Python, SQL, Excel, Power BI",
            )

            candidate_summary = st.text_area(
                "Candidate Summary",
                placeholder="Describe the candidate's background.",
            )

        st.divider()
        st.header("Job Requirements")

        col3, col4 = st.columns(2)

        with col3:
            job_title = st.text_input(
                "Job Title",
                placeholder="e.g. Junior Data Analyst",
            )

            job_experience = st.number_input(
                "Required Experience (years)",
                min_value=0.0,
                value=2.0,
                step=1.0,
            )

        with col4:
            job_required_skills = st.text_area(
                "Required Skills",
                placeholder="e.g. Python, SQL, data visualization",
            )

        job_description = st.text_area(
            "Job Description",
            placeholder="Describe the responsibilities and requirements.",
            height=120,
        )

        submitted = st.form_submit_button(
            "Analyze Candidate",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        required_fields = {
            "Candidate Role": candidate_role,
            "Candidate Education": candidate_education,
            "Candidate Skills": candidate_skills,
            "Candidate Summary": candidate_summary,
            "Job Title": job_title,
            "Required Skills": job_required_skills,
            "Job Description": job_description,
        }

        missing_fields = [
            name
            for name, value in required_fields.items()
            if not value.strip()
        ]

        if missing_fields:
            st.error(
                "Please complete: " + ", ".join(missing_fields)
            )

        else:
            # Match the exact text format used during preprocessing.
            
            combined_text = (
                "Candidate role: " + candidate_role + ". "
                + "Education: " + candidate_education + ". "
                + "Skills: " + candidate_skills + ". "
                + "Summary: " + candidate_summary + ". "
                + "Job title: " + job_title + ". "
                + "Job description: " + job_description + ". "
                + "Required skills: " + job_required_skills
            )

            experience_gap = (
                candidate_experience - job_experience
            )

            payload = {
                "combined_text": combined_text,
                "candidate_experience_years": candidate_experience,
                "job_required_experience": job_experience,
                "experience_gap": experience_gap,
            }

            try:
                with st.spinner("Analyzing candidate..."):
                    response = requests.post(
                        API_URL,
                        json=payload,
                        timeout=60,
                    )

                if response.status_code == 200:
                    result = response.json()

                    probability = float(
                        result["match_probability"]
                    )
                    prediction = int(result["prediction"])
                    result_label = result["result"]

                    # Save each successful prediction.

                    record = {
                        "timestamp": datetime.now().astimezone().isoformat(
                            timespec="seconds"
                        ),
                        "candidate_role": candidate_role,
                        "job_title": job_title,
                        "candidate_experience_years": (
                            candidate_experience
                        ),
                        "job_required_experience": job_experience,
                        "experience_gap": experience_gap,
                        "match_probability": probability,
                        "prediction": prediction,
                        "result": result_label,
                    }

                    try:
                        save_prediction(record)
                    except (OSError, PermissionError) as exc:
                        st.warning(
                            "Prediction succeeded, but its history "
                            "could not be saved."
                        )
                        st.caption(str(exc))

                    st.divider()
                    st.header("Candidate Match Results")

                    st.metric(
                        "Model Match Probability",
                        f"{probability:.1%}",
                    )

                    st.progress(
                        max(0.0, min(1.0, probability))
                    )

                    if prediction == 1:
                        st.success(
                            "Suitable Match — the model predicts "
                            "a positive match."
                        )
                    else:
                        st.warning(
                            "Potential Mismatch — the model predicts "
                            "a negative match."
                        )

                    st.caption(
                        "Model output is decision support, not a "
                        "verified measure of candidate suitability "
                        "or a final hiring recommendation."
                    )

                else:
                    st.error(
                        f"API error ({response.status_code}): "
                        f"{response.text}"
                    )

            except requests.RequestException as exc:
                st.error(
                    "Could not connect to FastAPI. Ensure the backend "
                    "is running at http://127.0.0.1:8000."
                )
                st.caption(str(exc))


# 6. TALENT ANALYTICS DASHBOARD

elif page == "Talent Analytics Dashboard":

    st.header("Talent Analytics Dashboard")
    st.write(
        "Review the predictions generated through this application."
    )

    history = load_history()

    if history.empty:
        st.info(
            "No prediction history yet. Open Candidate Matching, "
            "analyze a candidate, and return here to see the analytics."
        )

    else:
        history["match_probability"] = pd.to_numeric(
            history["match_probability"],
            errors="coerce",
        )

        history["prediction"] = pd.to_numeric(
            history["prediction"],
            errors="coerce",
        )

        history = history.dropna(
            subset=["match_probability", "prediction"]
        )

        total_candidates = len(history)

        suitable_count = int(
            (history["prediction"] == 1).sum()
        )

        mismatch_count = int(
            (history["prediction"] == 0).sum()
        )

        average_probability = history[
            "match_probability"
        ].mean()

        # KPI cards

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Candidates Analyzed",
            total_candidates,
        )

        col2.metric(
            "Suitable Matches",
            suitable_count,
        )

        col3.metric(
            "Potential Mismatches",
            mismatch_count,
        )

        col4.metric(
            "Average Match Probability",
            f"{average_probability:.1%}",
        )

        st.divider()

        # Prediction distribution chart

        st.subheader("Prediction Distribution")

        distribution = pd.DataFrame(
            {
                "Category": [
                    "Suitable Match",
                    "Potential Mismatch",
                ],
                "Candidates": [
                    suitable_count,
                    mismatch_count,
                ],
            }
        ).set_index("Category")

        st.bar_chart(distribution)

        # Match probability chart

        st.subheader("Match Probability by Analysis")

        probability_history = history[
            ["timestamp", "match_probability"]
        ].copy()

        probability_history["Analysis"] = range(
            1, len(probability_history) + 1
        )

        probability_chart = probability_history.set_index(
            "Analysis"
        )[["match_probability"]]

        st.line_chart(probability_chart)

        st.caption(
            "The probability chart shows model outputs in analysis "
            "order; it does not represent changes in model accuracy."
        )

        # Prediction history table

        st.subheader("Prediction History")

        display_columns = [
            "timestamp",
            "candidate_role",
            "job_title",
            "candidate_experience_years",
            "job_required_experience",
            "match_probability",
            "result",
        ]

        st.dataframe(
            history[display_columns].sort_values(
                "timestamp",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Download history

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction History (CSV)",
            data=csv_data,
            file_name="recruitment_prediction_history.csv",
            mime="text/csv",
        )

    st.caption(
        "Dashboard statistics are based only on saved predictions "
        "from this application, not the training dataset."
    )
