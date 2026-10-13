# AI-Powered Smart Recruitment & Talent Analytics System

## 📌 Project Overview

The **AI-Powered Smart Recruitment & Talent Analytics System** is a deep learning-based application designed to support recruitment decision-making through candidate and job requirement analysis.

The system uses a Long Short-Term Memory (LSTM) neural network to analyze candidate-related information and generate recruitment predictions. It combines a Streamlit frontend, a FastAPI backend, MLflow experiment tracking, and Docker containerization to provide an integrated application.

## 🎯 Objectives

- Develop a deep learning model for recruitment-related prediction.
- Analyze candidate information alongside job requirements.
- Use text and numerical features for model prediction.
- Track model training experiments using MLflow.
- Build an interactive frontend using Streamlit.
- Develop a REST API using FastAPI.
- Containerize the application using Docker and Docker Compose.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | Data preprocessing and evaluation utilities |
| TensorFlow / Keras | LSTM deep learning model |
| MLflow | Experiment tracking and model management |
| FastAPI | Backend API development |
| Streamlit | Interactive frontend |
| Docker | Application containerization |
| Docker Compose | Managing multiple services |
| SQLite | MLflow backend database |

## 🧠 Model Development

The project uses an LSTM (Long Short-Term Memory) neural network for recruitment-related prediction.

### Features

The model uses candidate and job-related information, including:

- Candidate experience
- Required job experience
- Experience gap
- Text-based recruitment information processed using a text vectorizer

### Preprocessing

The preprocessing workflow includes:

- Preparing text and numerical features.
- Transforming text information into numerical representations.
- Scaling numerical features.
- Preparing the processed features for model prediction.

The trained model and associated preprocessing artifacts are saved for reuse by the application.

## 📊 Model Evaluation and Experiment Tracking

MLflow is used to track model training experiments and review logged parameters, metrics, and other available run information.

The project includes model evaluation artifacts such as:

- `experiment_results.csv`
- `confusion_matrix.csv`

The metrics and evaluation results should be interpreted according to the actual training and evaluation outputs.

## 🏗️ Project Structure


AI-Smart-Recruitment-Talent-Analytics-System/
│
├── Backend/
│   ├── main.py
│   ├── schemas.py
│   └── requirements.txt
│
├── Frontend/
│   ├── app.py
│   └── requirements.txt
│
├── Data/
│
├── Notebook/
│
├── src/
│
├── models/
│   ├── recruitment_model.keras
│   ├── scaler.pkl
│   ├── model_metadata.json
│   ├── experiment_results.csv
│   └── confusion_matrix.csv
│
├── mlruns/
├── mlflow.db
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── README.md


*Note: The structure above highlights the main project components. Additional files may be present in individual folders.*

## ⚙️ Installation and Setup

### Prerequisites

Install or configure the following:

- Python
- Anaconda or another Python environment manager
- Docker Desktop
- Git (optional, for version control)

### 1. Clone the Repository

If the project is available on GitHub:

```bash
git clone <your-github-repository-url>
cd <your-project-folder>
```

Alternatively, open the existing project folder directly in VS Code.

### 2. Activate the Python Environment

If you are using the Conda environment named `venv`:

```bash
conda activate venv
```

### 3. Install Dependencies

Install the backend dependencies:

```bash
pip install -r Backend/requirements.txt
```

Install the frontend dependencies:

```bash
pip install -r Frontend/requirements.txt
```

## 🚀 Running the Application with Docker

Docker Compose runs the frontend and backend as separate containers.

### 1. Start Docker Desktop

Ensure that Docker Desktop is running.

### 2. Navigate to the Project Directory

Open the VS Code terminal in the project root folder.

### 3. Build and Start the Services

```bash
docker compose up --build -d
```

Use this command when a rebuild is required. To restart the existing containers without rebuilding, use:

```bash
docker compose up -d
```

### 4. Access the Application

**Streamlit Frontend**

http://localhost:8502

Use the frontend to enter recruitment-related information and view prediction results.

**FastAPI Backend Documentation**

http://localhost:8001/docs

Use the interactive API documentation to inspect and test the available endpoints.

### 5. Check Running Containers

```bash
docker compose ps
```

### 6. Stop the Application

```bash
docker compose down
```

This stops and removes the project's containers and Compose network without deleting the project source files or model artifacts stored on the host.

## 📈 Running MLflow

To open the MLflow dashboard locally, activate the appropriate Python environment and run:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open the dashboard at:

http://127.0.0.1:5000

MLflow can be used to inspect experiments and their recorded runs. The displayed information depends on the tracking database and artifacts configured by the training code.

Press `Ctrl + C` in the MLflow terminal to stop the dashboard server.

## 🔌 API Usage

The backend exposes API endpoints through FastAPI.

Open the API documentation at:

http://localhost:8001/docs

Use the interactive documentation to inspect the available endpoints, submit requests, and review the responses.

The frontend communicates with the backend through the Docker Compose service network when the application is running in Docker.

## 💾 Saved Model Artifacts

The `models/` directory contains the trained model and associated files used by the application.

- `recruitment_model.keras` — saved LSTM model.
- `scaler.pkl` — saved numerical scaling artifact.
- `model_metadata.json` — model-related metadata.
- `experiment_results.csv` — experiment evaluation results.
- `confusion_matrix.csv` — confusion matrix data.

## 🌟 Key Features

- Deep learning-based recruitment prediction.
- Candidate and job requirement feature analysis.
- Text and numerical feature processing.
- Interactive Streamlit user interface.
- FastAPI backend integration.
- MLflow experiment tracking.
- Docker-based deployment.
- Saved model artifacts for inference.

## 🔮 Future Enhancements

- Integrate additional candidate and job-related features.
- Improve model evaluation and validation.
- Add explainability features to help interpret predictions.
- Enhance prediction history and reporting.
- Implement appropriate fairness checks and human review for recruitment decisions.
- Deploy the application to a cloud platform.

## ⚠️ Responsible Use

Recruitment predictions should be treated as decision-support outputs rather than final hiring decisions. Model predictions may contain errors or reflect biases in the training data. Candidate evaluation should include human review, appropriate privacy safeguards, and fairness assessments.
