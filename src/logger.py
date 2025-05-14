import logging


PRINT_LOG = True

handlers = [logging.FileHandler("../data/chat_log.txt", encoding="utf-8")]
if PRINT_LOG:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers
)

logger = logging.getLogger("jarvis")
