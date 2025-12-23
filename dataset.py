from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm

class SampleInfo(BaseModel):
    path: Path
    metadata: dict

class DatasetInfo(BaseModel):
    samples: list[SampleInfo]
    name: str

class DatasetBase:
    def get_dataset_info(self) -> DatasetInfo:
        pass

class CatBreeds(DatasetBase):
    def __init__(self, dataset_path: str, dataset_name: str):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
    
    def get_dataset_info(self):
        paths = []
        path = Path(self.dataset_path)
        for folder in tqdm(path.iterdir()):
            img_files = folder.glob("**/*.jpg")
            for img_file in img_files:
                paths.append(SampleInfo(path=img_file, metadata={"Breeds":folder.name}))
        return DatasetInfo(samples=paths, name=self.dataset_name)
