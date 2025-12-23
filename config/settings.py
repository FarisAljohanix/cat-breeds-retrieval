from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent 

class Settings(BaseSettings):
    model_config = SettingsConfigDict(yaml_file=ROOT_DIR / "config.yaml")
    qdrant_host: str
    qdrant_port: int

    @classmethod
    def settings_customise_sources(cls, settings_cls: type[BaseSettings], *args, **kwargs):
        return (YamlConfigSettingsSource(settings_cls), )
    
settings = Settings()