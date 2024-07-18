import logging
from logging.handlers import TimedRotatingFileHandler
import os
import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s |%(levelname)s| %(filename)s.%(funcName)s - %(threadName)s: %(message)s')

# add formatter to handler
console = logging.StreamHandler()
console.setFormatter(formatter)


# File rollback
os.makedirs("./logs", exist_ok=True)
file_handler = logging.FileHandler("./logs/langchain-agents.log")
file_handler.setFormatter(formatter)
# max_bytes = 1024 * 1024 * 250
# file_handler.maxBytes = max_bytes

log_folder = "./logs/" + datetime.datetime.now().strftime("%Y-%m")
os.makedirs(log_folder, exist_ok=True)
log_file_daily = os.path.join(log_folder, "langchain-agents.log")

file_handler_daily = TimedRotatingFileHandler(
    log_file_daily, when="midnight", interval=1, backupCount=30)
file_handler_daily.setFormatter(formatter)


# add handlers to logger
logger.addHandler(console)
logger.addHandler(file_handler)
logger.addHandler(file_handler_daily)