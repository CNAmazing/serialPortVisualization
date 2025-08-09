import logging

class EmojiFormatter(logging.Formatter):
    def format(self, record):
        level_to_emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "💥"
        }
        record.levelname = f"{level_to_emoji.get(record.levelname, '')} {record.levelname}"
        return super().format(record)

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(EmojiFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("This log has an emoji!")