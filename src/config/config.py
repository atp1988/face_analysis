# import os
# from dotenv import load_dotenv
# from pathlib import Path
# import json

# # Load .env file
# env_path = Path('/app/.env')
# load_dotenv(dotenv_path=env_path)

# class Settings:
#     """Configuration settings loaded from .env file."""
#     SQL_DRIVER = os.getenv("SQL_DRIVER")
#     SQL_SERVER = os.getenv("SQL_SERVER")
#     SQL_DATABASE = os.getenv("SQL_DATABASE")
#     SQL_UID = os.getenv("SQL_UID")
#     SQL_PWD = os.getenv("SQL_PWD")
#     CAMERAS = json.loads(os.getenv("CAMERAS", "{}"))

# settings = Settings()


## src/config/config.py:
import os
from dotenv import load_dotenv
from pathlib import Path
import json

# Load .env file
env_path = Path('/app/.env')
load_dotenv(dotenv_path=env_path)

class Settings:
    """Configuration settings loaded from .env file."""
    # SQL Server Settings
    SQL_DRIVER = os.getenv("SQL_DRIVER")
    SQL_SERVER = os.getenv("SQL_SERVER")
    SQL_DATABASE = os.getenv("SQL_DATABASE")
    SQL_UID = os.getenv("SQL_UID")
    SQL_PWD = os.getenv("SQL_PWD")
    
    # Camera RTSP Streams
    CAMERAS = json.loads(os.getenv("CAMERAS", "{}"))

    # --- ADDED: Redis Settings ---
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


settings = Settings()