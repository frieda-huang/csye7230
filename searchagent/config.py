from pydantic import Field
from pydantic_settings import BaseSettings, SettingConfigDict


class Settings(BaseSettings):
    model_config = SettingConfigDict(env_file=".env", env_file_encoding="utf-8")
    model_name = Field(
        default="vidore/colpali-v1.2",
        description="Colpali is a document retrieval model.",
    )
    benchmark_dataset_name = Field(
        default="vidore/syntheticDocQA_artificial_intelligence_test",
        description="This dataset includes documents about Artificial Intelligence.",
    )
    seed_user_password = Field(description="Password to seed a dummy user")
    hf_api_key: str
    database_url: str


settings = Settings()
