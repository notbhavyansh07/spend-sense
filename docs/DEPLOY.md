# SpendSense Deployment Guide

This guide explains how to run SpendSense locally, via Docker, and how to deploy to Streamlit Community Cloud.

> **Note on Cloud Deployment Status**: Streamlit Community Cloud deployment instructions below are documented based on platform specifications, but the live cloud deployment itself is **untested** in this local development environment.

---

## 1. Local Run (Fastest for Viva Demonstrations)

1. Ensure Python 3.12 is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate the labeled training dataset and train the model:
   ```bash
   python data/generate_dataset.py
   python main.py train
   ```
4. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
   The dashboard will automatically open at `http://localhost:8501`.
5. In the sidebar, click **Load 12-Month Demo Data** to populate the charts, forecast models, and anomaly tables.

---

## 2. Running Backend and Dashboard with Docker

Docker compose coordinates both the FastAPI backend on port 8000 and the Streamlit frontend on port 8501.

1. Build and run containers:
   ```bash
   docker compose up --build
   ```
2. Interactive Swagger API documentation: `http://localhost:8000/docs`
3. Frontend Dashboard: `http://localhost:8501`

---

## 3. Deploying to Streamlit Community Cloud (Untested)

To deploy the standalone dashboard to Streamlit Community Cloud:

1. Push your repository to GitHub:
   ```bash
   git push origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in using your GitHub account.
3. Click **New app**.
4. Configure application settings:
   - **Repository**: `notbhavyansh07/spend-sense`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy**.
6. On initial boot, `app.py` checks if `models/categorizer.joblib` exists. Because trained models are excluded by `.gitignore`, `app.py` automatically runs `data/generate_dataset.py` and `categorizer.train()` on first startup.
7. Open the generated public URL and click **Load 12-Month Demo Data** to initialize records. Note that ephemeral container restarts will reset local SQLite storage.
