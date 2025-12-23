from typing import Optional
from PIL import Image
import numpy as np
import torch
import open_clip


class OpenCLIPEmbeddingBackend:

    def __init__(
        self,
        model_name: str,
        dataset: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 128,
    ) -> None:
        self.device = device
        if dataset is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, device=device
            )
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=dataset, device=device
            )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.batch_size = batch_size

    def compute_image_embedding(self, image: Image.Image) -> np.ndarray:
        image = self.preprocess(image).unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        image_features = image_features.cpu().detach().numpy().squeeze()
        return image_features








