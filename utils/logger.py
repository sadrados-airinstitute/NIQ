import logging
import os

class Logger:
    def __init__(self, log_dir: str, log_file: str = "api.log"):
        """
        Initializes the Logger object to log to both a file and console.
        
        Args:
            log_dir (str): The directory where the log file will be saved.
            log_file (str): The name of the log file (default is 'training.log').
        """
        self.log_dir = log_dir
        self.log_file = log_file
        self.logger = logging.getLogger(log_file.split("/")[-1])
        self.setup_logger()

    def setup_logger(self):
        """
        Set up the logger to write logs to a file and console.
        """
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Define the log file path
        log_path = os.path.join(self.log_dir, self.log_file)
        
        # Set the logger level
        self.logger.setLevel(logging.INFO)

        # Create a file handler to write logs to the file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        
        # Create a console handler to log to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Set up a formatter for clear logs
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        """
        Returns the logger object.
        
        Returns:
            logger (logging.Logger): The configured logger.
        """
        return self.logger