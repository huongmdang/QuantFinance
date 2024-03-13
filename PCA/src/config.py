"""This module extracts information from `.env` file so that
we can use username, password, database name and api key in other parts.
"""

import os
from pydantic_settings import BaseSettings

def return_full_path(filename: str = ".env") -> str:
    """Uses os to return the correct path of the `.env` file."""
    absolute_path = os.path.abspath(__file__)
    directory_name = os.path.dirname(absolute_path)
    full_path = os.path.join(directory_name, filename)
    return full_path

class Settings(BaseSettings):
    """Uses pydantic to define settings for project."""

    hostname: str
    dbname: str
    uname: str
    pwd: str
        
    class Config:
        env_file = return_full_path(".env")

settings = Settings()


