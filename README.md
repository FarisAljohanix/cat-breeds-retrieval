## Cat Image Retrieval (Streamlit + Qdrant)


Uploading Screen Recording 2025-12-23 at 6.41.17 PM.mov…


A simple image-retrieval demo that embeds cat images with OpenCLIP, stores them in Qdrant, and lets you query by uploading a new image.

### Prerequisites
- Python 3.12+
- Docker (for Qdrant)
- `uv` (recommended) or `pip`

### Setup
```bash
cd /path/to/CLIP

# Install deps (uv)
uv sync

# If using pip instead:
# python -m venv .venv
# source .venv/bin/activate
# python -m pip install -e .
```

### Configure Qdrant
Edit `config/config.yaml` if you change host/port. Default expects Qdrant on `localhost:6331`.

Start Qdrant with the provided data volume:
```bash
docker compose up -d    # uses docker-compose.yaml in this folder
```

### Run the app
```bash
# activate env if needed
source .venv/bin/activate   # or let uv handle it per command

python -m streamlit run main.py
# or: uv run streamlit run main.py
```
Open the shown URL (default http://localhost:8501), upload a cat image, and view the retrieved results.

### Stop services
```bash
docker compose down   # stop Qdrant
```
