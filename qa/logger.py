import sys
import logging


class QALogger:
    # TODO
    #  Make this class as a singleton

    DEBUG_COLOR = "\x1b[30;20m"
    INFO_COLOR = "\x1b[34;20m"
    WARNING_COLOR = "\x1b[33;20m"
    ERROR_COLOR = "\x1b[31;20m"
    CRITICAL_COLOR = "\x1b[1;41m"
    RESET_COLOR = "\x1b[0m"

    FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s'

    @staticmethod
    def get_logger(name: str = __name__) -> logging.Logger:
        logging.addLevelName(
            level=logging.DEBUG,
            levelName=f"{QALogger.DEBUG_COLOR}{logging.getLevelName(logging.DEBUG)}{QALogger.RESET_COLOR}"
        )
        logging.addLevelName(
            level=logging.INFO,
            levelName=f"{QALogger.INFO_COLOR}{logging.getLevelName(logging.INFO)}{QALogger.RESET_COLOR}"
        )
        logging.addLevelName(
            level=logging.WARNING,
            levelName=f"{QALogger.WARNING_COLOR}{logging.getLevelName(logging.WARNING)}{QALogger.RESET_COLOR}"
        )
        logging.addLevelName(
            level=logging.ERROR,
            levelName=f"{QALogger.ERROR_COLOR}{logging.getLevelName(logging.ERROR)}{QALogger.RESET_COLOR}"
        )
        logging.addLevelName(
            level=logging.CRITICAL,
            levelName=f"{QALogger.CRITICAL_COLOR}{logging.getLevelName(logging.CRITICAL)}{QALogger.RESET_COLOR}"
        )

        _logger = logging.getLogger(name)
        _logger.setLevel(logging.DEBUG)

        log_formatter = logging.Formatter(QALogger.FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.DEBUG)

        if _logger.handlers:
            _logger.handlers = []

        _logger.addHandler(console_handler)

        return _logger


if __name__ == "__main__":
    logger = QALogger.get_logger()
    logger.debug("Used for debugging your code.")
    logger.info("Informative messages from your code.")
    logger.warning("Everything works but there is something to be aware of.")
    logger.error("There's been a mistake with the process.")
    logger.critical("There is something terribly wrong and process may terminate.")
