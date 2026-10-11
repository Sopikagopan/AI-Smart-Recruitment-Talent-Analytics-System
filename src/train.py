from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# PROJECT PATHS AND SETTINGS

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "recruitment_processed.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = (ROOT / "mlflow.db").as_posix()

EXPERIMENT_NAME = "Smart_Recruitment_LSTM"
REGISTERED_MODEL_NAME = "SmartRecruitmentLSTM"

EPOCH_EXPERIMENTS = [5, 10, 15, 20]
BATCH_SIZE = 32
RANDOM_STATE = 42

MAX_TOKENS = 10000
SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 64
LSTM_UNITS = 64

TEXT_COL = "combined_text"

NUMERIC_COLS = [
    "candidate_experience_years",
    "job_required_experience",
    "experience_gap",
]

TARGET_COL = "match_label"


# REPRODUCIBILITY

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# MLFLOW CONFIGURATION

mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH}")
mlflow.set_experiment(EXPERIMENT_NAME)


# LOAD AND VALIDATE DATASET

print("\nLoading processed recruitment dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run src/preprocess.py first."
    )

df = pd.read_csv(DATA_PATH)

required_columns = [TEXT_COL] + NUMERIC_COLS + [TARGET_COL]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df[required_columns].copy()

df = df.dropna(subset=required_columns).reset_index(drop=True)

df[TEXT_COL] = df[TEXT_COL].astype(str)

for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="raise")

df[TARGET_COL] = pd.to_numeric(
    df[TARGET_COL], errors="raise"
).astype(int)

if set(df[TARGET_COL].unique()) != {0, 1}:
    raise ValueError(
        "match_label must contain both binary labels 0 and 1."
    )

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df[TARGET_COL].value_counts())


# TRAIN, VALIDATION AND TEST SPLITS

train_df, temp_df = train_test_split(
    df,
    test_size=0.36,
    random_state=RANDOM_STATE,
    stratify=df[TARGET_COL],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=(20 / 36),
    random_state=RANDOM_STATE,
    stratify=temp_df[TARGET_COL],
)

y_train = train_df[TARGET_COL].to_numpy()
y_val = val_df[TARGET_COL].to_numpy()
y_test = test_df[TARGET_COL].to_numpy()

print("\nData split completed:")
print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Testing rows:", len(test_df))


# NUMERICAL FEATURE SCALING

scaler = StandardScaler()

X_train_num = scaler.fit_transform(
    train_df[NUMERIC_COLS]
).astype(np.float32)

X_val_num = scaler.transform(
    val_df[NUMERIC_COLS]
).astype(np.float32)

X_test_num = scaler.transform(
    test_df[NUMERIC_COLS]
).astype(np.float32)


# TEXT PREPROCESSING

text_vectorizer = keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
)

# Learn vocabulary from training text only.

text_vectorizer.adapt(
    train_df[TEXT_COL].to_numpy(dtype=str)
)


X_train_text = train_df[TEXT_COL].astype(str).tolist()
X_val_text = val_df[TEXT_COL].astype(str).tolist()
X_test_text = test_df[TEXT_COL].astype(str).tolist()



train_inputs = {
    "combined_text": tf.constant(
        X_train_text, dtype=tf.string
    ),
    "numeric_features": X_train_num,
}

val_inputs = {
    "combined_text": tf.constant(
        X_val_text, dtype=tf.string
    ),
    "numeric_features": X_val_num,
}

test_inputs = {
    "combined_text": tf.constant(
        X_test_text, dtype=tf.string
    ),
    "numeric_features": X_test_num,
}



# 8. LSTM MODEL-BUILDING FUNCTION

def build_lstm_model():

    text_input = keras.Input(
        shape=(),
        dtype=tf.string,
        name="combined_text",
    )

    numeric_input = keras.Input(
        shape=(len(NUMERIC_COLS),),
        dtype=tf.float32,
        name="numeric_features",
    )

    tokens = text_vectorizer(text_input)

    embedding = keras.layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="embedding",
    )(tokens)

    lstm_output = keras.layers.LSTM(
        LSTM_UNITS,
        name="lstm_layer",
    )(embedding)

    combined = keras.layers.Concatenate()(
        [lstm_output, numeric_input]
    )

    hidden = keras.layers.Dense(
        32,
        activation="relu",
        name="dense_layer",
    )(combined)

    hidden = keras.layers.Dropout(0.3)(hidden)

    output = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="match_probability",
    )(hidden)

    model = keras.Model(
        inputs={
            "combined_text": text_input,
            "numeric_features": numeric_input,
        },
        outputs=output,
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# RUN AND TRACK EPOCH EXPERIMENTS

experiment_results = []
best_model = None
best_run_id = None
best_val_f1 = -1.0
best_epoch_setting = None

for epochs in EPOCH_EXPERIMENTS:

    print("\n" + "=" * 60)
    print(f"STARTING LSTM EXPERIMENT: {epochs} EPOCHS")
    print("=" * 60)

    # Build a fresh model for every experiment.

    keras.backend.clear_session()
    tf.random.set_seed(RANDOM_STATE)

    model = build_lstm_model()

    # Create a new callback for each independent run.

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    with mlflow.start_run(
        run_name=f"LSTM_{epochs}_epochs"
    ) as run:

        run_id = run.info.run_id

        # Log experiment parameters.

        mlflow.log_params({
            "model_type": "LSTM",
            "epochs_max": epochs,
            "batch_size": BATCH_SIZE,
            "random_state": RANDOM_STATE,
            "max_tokens": MAX_TOKENS,
            "sequence_length": SEQUENCE_LENGTH,
            "embedding_dim": EMBEDDING_DIM,
            "lstm_units": LSTM_UNITS,
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "early_stopping": True,
            "early_stopping_patience": 3,
            "validation_split": 0.16,
            "test_split": 0.20,
        })

        # Train the model.

        history = model.fit(
            train_inputs,
            y_train,
            validation_data=(val_inputs, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=1,
        )

        actual_epochs = len(history.history["loss"])

        # Log training history for comparison.

        for epoch_index in range(actual_epochs):

            mlflow.log_metrics(
                {
                    "train_loss": float(
                        history.history["loss"][epoch_index]
                    ),
                    "train_accuracy": float(
                        history.history["accuracy"][epoch_index]
                    ),
                    "val_loss": float(
                        history.history["val_loss"][epoch_index]
                    ),
                    "val_accuracy": float(
                        history.history["val_accuracy"][epoch_index]
                    ),
                },
                step=epoch_index + 1,
            )

        # Evaluate on validation data to select the model.

        val_probabilities = model.predict(
            val_inputs, verbose=0
        ).ravel()

        val_predictions = (
            val_probabilities >= 0.5
        ).astype(int)

        val_accuracy = accuracy_score(
            y_val, val_predictions
        )

        val_precision = precision_score(
            y_val, val_predictions, zero_division=0
        )

        val_recall = recall_score(
            y_val, val_predictions, zero_division=0
        )

        val_f1 = f1_score(
            y_val, val_predictions, zero_division=0
        )

        best_val_loss = min(history.history["val_loss"])

        mlflow.log_metrics({
            "val_accuracy_final": float(val_accuracy),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
            "best_val_loss": float(best_val_loss),
            "epochs_completed": actual_epochs,
        })

        # Log this experiment's model.

        mlflow.tensorflow.log_model(
            model,
            artifact_path="lstm_model",
        )

        experiment_results.append({
            "epochs_max": epochs,
            "epochs_completed": actual_epochs,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "best_val_loss": best_val_loss,
            "run_id": run_id,
        })

        print(f"\nMaximum epochs: {epochs}")
        print(f"Epochs completed: {actual_epochs}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation precision: {val_precision:.4f}")
        print(f"Validation recall: {val_recall:.4f}")
        print(f"Validation F1-score: {val_f1:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")

        # Select the best model using validation F1-score.

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_run_id = run_id
            best_epoch_setting = epochs

# COMPARE EXPERIMENT RESULTS

results_df = pd.DataFrame(experiment_results)

results_df = results_df.sort_values(
    by=["val_f1", "best_val_loss"],
    ascending=[False, True],
).reset_index(drop=True)

results_path = MODEL_DIR / "experiment_results.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("LSTM EXPERIMENT COMPARISON")
print("=" * 60)

print(
    results_df.drop(columns=["run_id"]).to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nBest maximum-epoch setting:", best_epoch_setting)
print(f"Best validation F1-score: {best_val_f1:.4f}")
print("Best MLflow run ID:", best_run_id)


# FINAL TEST EVALUATION OF SELECTED MODEL

print("\nEvaluating the selected model on the test set...")

test_probabilities = best_model.predict(
    test_inputs, verbose=0
).ravel()

test_predictions = (
    test_probabilities >= 0.5
).astype(int)

test_accuracy = accuracy_score(
    y_test, test_predictions
)

test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)

test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)

test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

print("\nFINAL TEST METRICS")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-score : {test_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    )
)

cm = confusion_matrix(y_test, test_predictions)

print("Confusion Matrix:")
print(cm)

cm_path = MODEL_DIR / "confusion_matrix.csv"

pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"],
).to_csv(cm_path)


# Log final test metrics to the selected experiment run.

with mlflow.start_run(run_id=best_run_id):

    mlflow.log_metrics({
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
    })

    mlflow.log_artifact(str(results_path))
    mlflow.log_artifact(str(cm_path))


# SAVE THE SELECTED MODEL AND PREPROCESSOR

best_model_path = MODEL_DIR / "recruitment_model.keras"
scaler_path = MODEL_DIR / "scaler.pkl"
metadata_path = MODEL_DIR / "model_metadata.json"

best_model.save(best_model_path)
joblib.dump(scaler, scaler_path)

metadata = {
    "model_type": "LSTM",
    "experiment_name": EXPERIMENT_NAME,
    "selected_run_id": best_run_id,
    "selected_max_epochs": best_epoch_setting,
    "best_validation_f1": float(best_val_f1),
    "test_accuracy": float(test_accuracy),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),
    "text_column": TEXT_COL,
    "numeric_columns": NUMERIC_COLS,
    "target_column": TARGET_COL,
    "threshold": 0.5,
    "max_tokens": MAX_TOKENS,
    "sequence_length": SEQUENCE_LENGTH,
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

# Save the shared preprocessing artifacts in the best run.
with mlflow.start_run(run_id=best_run_id):

    mlflow.log_artifact(str(scaler_path))
    mlflow.log_artifact(str(metadata_path))
    mlflow.log_artifact(str(best_model_path))

    mlflow.set_tag(
        "selected_as_best",
        "true",
    )


# REGISTER THE SELECTED MODEL IN MLFLOW

print("\nRegistering the selected model in MLflow...")

model_uri = f"runs:/{best_run_id}/lstm_model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME,
)

print("Registered model:", REGISTERED_MODEL_NAME)
print("Registered model version:", registered_model.version)


# COMPLETION SUMMARY

print("\n" + "=" * 60)
print("TRAINING AND EXPERIMENT TRACKING COMPLETED!")
print("=" * 60)

print("MLflow experiment:", EXPERIMENT_NAME)
print("Maximum epoch settings tested:", EPOCH_EXPERIMENTS)
print("Selected maximum-epoch setting:", best_epoch_setting)
print("Selected MLflow run:", best_run_id)
print("Registered model:", REGISTERED_MODEL_NAME)
print("Saved model:", best_model_path)
print("Saved scaler:", scaler_path)
print("Saved metadata:", metadata_path)
print("Experiment comparison:", results_path)
print("Confusion matrix:", cm_path)

print("\nNext step: open the MLflow UI to compare the runs.")
