# Week 12 Viva Notes: Testing, Docker, and Production Readiness

### 1. How does Docker ensure environment consistency between development and production?
Docker packages the Python 3.12 runtime, dependencies (`requirements.txt`), application code, and pre-trained model weights into an immutable container image built from `python:3.12-slim`. By defining multi-container orchestration in `docker-compose.yml`, both the FastAPI backend and Streamlit dashboard run in isolated networks with explicit port mappings (`8000:8000` and `8501:8501`) and named persistent volumes (`dbdata:/app/db`), preventing platform-dependent "works on my machine" failures.

### 2. Why does the Docker build run generate_dataset.py and main.py train during the build phase?
Because `.gitignore` prevents trained models from being tracked in git, building the Docker image without training would produce an image with an empty `models/` directory. By executing `RUN python data/generate_dataset.py && python main.py train` inside the `Dockerfile`, the container image is baked with pre-compiled model artifacts (`models/categorizer.joblib`), ensuring instantaneous startup when containers launch in production.

### 3. How does named volume persistence protect SQLite databases in containerized deployments?
Containers are ephemeral: when a container is restarted or updated, its writeable container layer is destroyed. By mapping `dbdata:/app/db` and pointing `SPENDSENSE_DB=/app/db/expenses.db`, SQLite database writes are routed to the host machine's persistent Docker volume storage. Data remains safe across container restarts, image updates, and redeployments.

### 4. What are the key architectural limitations of SpendSense?
1. **Synthetic Training Data**: Categories are learned from template-generated patterns with injected typos; while effective for baseline demonstration, real student receipts contain informal dialect and novel brand names that reduce out-of-distribution accuracy to ~60%.
2. **Short Forecasting Horizon**: Forecasting requires at least 6 months of history and is evaluated on 12 months, limiting seasonal modeling capabilities.
3. **Regex-Based SMS Extraction**: SMS parsing relies on deterministic pattern matching tailored to Indian banking conventions; unconventional bank notification formats require pattern updates.
4. **Constrained NLP Intent Whitelist**: The chat system supports 5 discrete aggregation intents, intentionally disallowing freeform SQL or multi-turn conversational dialog for security.

### 5. What are the natural future enhancements for SpendSense?
- Incorporating fine-tuned open-source small language models (SLMs) on device for fuzzy intent classification.
- Integrating account aggregator APIs (such as India's RBI Account Aggregator framework) to ingest real authenticated bank feeds directly with user consent.
- Implementing recurring subscription detection and alert push notifications for anomalous debits.

---

### One Thing That Could Break
If the Docker host runs out of disk space while building the container image, or if network connectivity drops while pulling base images from Docker Hub, `docker compose up --build` will terminate with an unhandled exit error during the `pip install` step.
