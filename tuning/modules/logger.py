import logging
import os
import yaml

# Define log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def load_config():
    """Load config.yaml file"""
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        return "INFO"  # Default log level if config doesn't exist

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config.get("logging", {}).get("log_level", "INFO")

def setup_logger(name="FlashTuning"):
    """Setup and configure logger"""
    logger = logging.getLogger(name)

    # Get log level from config.yaml
    log_level = load_config()
    logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))

    # Create console handler if not already added
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        # Add formatter with prefix
        formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# Create global logger instance
logger = setup_logger()