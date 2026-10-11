
import pandas as pd
from pathlib import Path


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "smart_recruitment_dl_dataset_5000.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recruitment_processed.csv"
)


# 2. Load the raw dataset
df = pd.read_csv(RAW_DATA_PATH)

print("Original dataset shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())


# 3. Remove duplicate rows
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows found:", duplicate_count)

df = df.drop_duplicates().copy()


# 4. Check missing values
print("\nMissing values before preprocessing:")
print(df.isnull().sum())


# 5. Create combined candidate-job text
text_columns = [
    "candidate_role",
    "candidate_education",
    "candidate_skills",
    "candidate_summary",
    "job_title",
    "job_description",
    "job_required_skills",
]

df[text_columns] = df[text_columns].fillna("").astype(str)

df["combined_text"] = (
    "Candidate role: " + df["candidate_role"] + ". "
    + "Education: " + df["candidate_education"] + ". "
    + "Skills: " + df["candidate_skills"] + ". "
    + "Summary: " + df["candidate_summary"] + ". "
    + "Job title: " + df["job_title"] + ". "
    + "Job description: " + df["job_description"] + ". "
    + "Required skills: " + df["job_required_skills"]
)


# 6. Prepare numerical features
df["candidate_experience_years"] = pd.to_numeric(
    df["candidate_experience_years"], errors="coerce"
)

df["job_required_experience"] = pd.to_numeric(
    df["job_required_experience"], errors="coerce"
)

df["experience_gap"] = (
    df["candidate_experience_years"]
    - df["job_required_experience"]
)


# 7. Remove rows with missing essential values
df = df.dropna(
    subset=[
        "candidate_experience_years",
        "job_required_experience",
        "match_label",
        "combined_text",
    ]
).copy()


# 8. Select features and target
processed_df = df[
    [
        "combined_text",
        "candidate_experience_years",
        "job_required_experience",
        "experience_gap",
        "match_label",
    ]
].copy()


# 9. Ensure target labels are numeric
processed_df["match_label"] = pd.to_numeric(
    processed_df["match_label"], errors="raise"
).astype(int)


# 10. Save processed dataset
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

processed_df.to_csv(PROCESSED_DATA_PATH, index=False)


# 11. Display results
print("\nPreprocessing completed successfully!")
print("Processed dataset shape:", processed_df.shape)

print("\nProcessed dataset preview:")
print(processed_df.head())

print("\nMissing values in processed dataset:")
print(processed_df.isnull().sum())

print("\nTarget distribution:")
print(processed_df["match_label"].value_counts().sort_index())

print("\nSaved processed dataset to:")
print(PROCESSED_DATA_PATH)
