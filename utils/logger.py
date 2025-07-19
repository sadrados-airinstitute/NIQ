import logging

def setup_logger(log_dir, log_file="training.log"):
    """
    Setup the logger to write logs to a file and console.
    
    Args:
        log_dir (str): The directory where the log file will be saved.
        log_file (str): The name of the log file.
        
    Returns:
        logger (logging.Logger): Configured logger.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Define the log file path
    log_path = os.path.join(log_dir, log_file)
    
    # Create a logger
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)  # Set log level to INFO

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler to log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Set up a formatter to display logs in a clear format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger