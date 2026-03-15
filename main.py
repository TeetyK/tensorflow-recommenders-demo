import tensorflow as tf
import tensorflow_recommenders as tfrs
from src.train import train_eval
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
import os
from datetime import datetime

log_folder = "logs/app_logs"
os.makedirs(log_folder, exist_ok=True)

log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
log_path = os.path.join(log_folder, log_filename)

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_path),  
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("---  Recommender System ---")
if __name__ == "__main__":
    train_eval()