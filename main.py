import streamlit as st
from pathlib import Path
from io import BytesIO
from PIL import Image

from vector_db import QdrantVectorDB
from clip import OpenCLIPEmbeddingBackend
from config.settings import settings

COLLECTION_NAME = "Cat Dataset"
TOP_K = 12
ROOT_DIR = Path(__file__).resolve().parent

embedding_model = OpenCLIPEmbeddingBackend(model_name="ViT-B-16-SigLIP-512", dataset="webli")
qdrant_vector_db = QdrantVectorDB(
    collection_name=COLLECTION_NAME,
    embedding_dim=768,
    host=settings.qdrant_host,
    port=settings.qdrant_port,
)

st.set_page_config(page_title="Cat Retrieval", layout="wide")
st.title("Cat Image Retrieval")
img_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp"])
retrieve_clicked = st.button("Retrieve Cat Image")


def resolve_image_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None

    prefix = "/home/faris/LLMs/CLIP"
    if raw_path.startswith(prefix):
        tail = raw_path[len(prefix):].lstrip("/")
        candidate = ROOT_DIR / tail
        if candidate.exists():
            return candidate

    # Direct string replacement of the old prefix in case subtle differences prevented the startswith branch.
    replaced = Path(raw_path.replace(prefix, str(ROOT_DIR)))
    if replaced.exists():
        return replaced

    p = Path(raw_path)
    if p.exists():
        return p

    # If the stored path is relative, resolve it from the project root.
    if not p.is_absolute():
        candidate = ROOT_DIR / p
        if candidate.exists():
            return candidate

    # If the path contains the dataset folder, rebuild it under the local project root.
    parts = p.parts
    if "cat breeds" in parts:
        subpath = Path(*parts[parts.index("cat breeds"):])
        candidate = ROOT_DIR / subpath
        if candidate.exists():
            return candidate

    return None


def show_results(points):
    if not points:
        st.info("No results returned.")
        return

    placeholder_buf = BytesIO()
    Image.new("RGB", (256, 256), color="#222222").save(placeholder_buf, format="PNG")
    placeholder_bytes = placeholder_buf.getvalue()

    cols = st.columns(min(4, len(points)))
    for idx, point in enumerate(points):
        payload = point.payload or {}
        path = payload.get("path")
        metadata = payload.get("metadata", {})
        breed = ""
        if isinstance(metadata, dict):
            breed = metadata.get("breed") or metadata.get("Breeds") or ""

        col = cols[idx % len(cols)]
        img_bytes = None
        resolved_path = resolve_image_path(path) if path else None
        error_note = ""
        if resolved_path and resolved_path.exists():
            try:
                # Ensure the file is non-empty and readable by PIL.
                if resolved_path.stat().st_size > 0:
                    with open(resolved_path, "rb") as f:
                        Image.open(f).verify()  # validate image header
                        f.seek(0)
                        img_bytes = f.read()
                else:
                    error_note = " (empty file)"
            except Exception:
                error_note = " (unreadable image)"

        if not img_bytes:
            img_bytes = placeholder_bytes

        col.image(img_bytes, caption=None, width="stretch")
        breed_line = f"Breed: {breed}" if breed else "Breed: unknown"
        score_line = f"Score: {point.score:.4f}"
        extra = f"{error_note}".strip()
        if extra:
            col.markdown(f"{breed_line}  \n{score_line}  \n{extra}")
        else:
            col.markdown(f"{breed_line}  \n{score_line}")


if retrieve_clicked and img_file:
    query_image = Image.open(img_file)
    embedding = embedding_model.compute_image_embedding(query_image)
    results = qdrant_vector_db.search(embedding, top_k=TOP_K)

    st.subheader("Query image")
    st.image(query_image, caption="Uploaded image", width="stretch")

    st.subheader("Retrieved images")
    show_results(results)
elif retrieve_clicked and not img_file:
    st.warning("Please upload an image first.")
