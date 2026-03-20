
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://pse:password@localhost:5432/pse"

    # Zarr / array store
    zarr_store_path: str = "/data/zarr"
    zarr_store_url: str | None = None  # e.g. s3://bucket/zarr

    # Data source credentials
    era5_cds_api_key: str | None = None
    era5_cds_api_url: str = "https://cds.climate.copernicus.eu/api"
    copernicus_dataspace_client_id: str | None = None
    copernicus_dataspace_client_secret: str | None = None

    # Auth
    jwt_secret_key: str = "change-me-in-production"
    api_key_salt: str = "change-me-in-production"

    # App
    environment: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
