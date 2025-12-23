from dataset import CatBreeds, DatasetInfo
from vector_db import QdrantVectorDB
from config.settings import settings
from PIL import Image
from clip import OpenCLIPEmbeddingBackend
from tqdm import tqdm
from config.logging import setup_logging

logger = setup_logging(__name__)

def get_dataset() -> DatasetInfo:
    cat_dataset = CatBreeds(dataset_path="/Users/krozsad/Desktop/scrpits/CLIP/cat breeds/cat-breeds/cat-breeds", dataset_name="Cat_Dataset")
    return cat_dataset.get_dataset_info()

def main():
    collection_name = "Cat Dataset"
    embedding_dim = 768

    embedding_model = OpenCLIPEmbeddingBackend(model_name="ViT-B-16-SigLIP-512", dataset="webli")
    qdrant_vector_db = QdrantVectorDB(collection_name=collection_name, embedding_dim=embedding_dim, host=settings.qdrant_host, port=settings.qdrant_port)
    qdrant_vector_db.create_collection(recreate=True)
    dataset_info = get_dataset()

    total_samples = len(dataset_info.samples)
    with tqdm(total=total_samples, desc="Indexing all datasets") as pbar:
        for sample in dataset_info.samples:
            image = Image.open(sample.path)
            embedding = embedding_model.compute_image_embedding(image)
            payload = {
                "path": sample.path,
                "metadata": sample.metadata,
            }
            qdrant_vector_db.insert(embedding, payload)
            pbar.update(1)

if __name__=="__main__":
    main()
