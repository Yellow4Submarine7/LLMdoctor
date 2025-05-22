import logging
import sys

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logger(name: str, level: str = None, log_file: str = None):
    """
    Sets up a logger for consistent logging across the project.

    Args:
        name (str): The name of the logger (e.g., __name__ from the calling module).
        level (str, optional): The logging level (e.g., "INFO", "DEBUG"). 
                               Defaults to DEFAULT_LOG_LEVEL or "INFO".
        log_file (str, optional): Path to a file to save logs. 
                                  If None, logs only to console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if level is None:
        level = DEFAULT_LOG_LEVEL
        
    logger_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(logger_level)

    # Prevent duplicate handlers if logger was already configured elsewhere (e.g. by root logger)
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(console_handler)

        # File Handler (optional)
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, mode='a') # Append mode
                file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
                logger.addHandler(file_handler)
                logger.info(f"Logging to file: {log_file}")
            except Exception as e:
                logger.error(f"Failed to set up log file at {log_file}: {e}")
                
    return logger

if __name__ == '__main__':
    # Example usage of the logger
    
    # Logger that only logs to console
    logger_console = setup_logger("ConsoleLoggerTest", level="DEBUG")
    logger_console.debug("This is a debug message for console.")
    logger_console.info("This is an info message for console.")
    logger_console.warning("This is a warning message for console.")

    # Logger that logs to console and file
    # Create a dummy log file for testing
    dummy_log_file = "dummy_test_log.log"
    logger_file = setup_logger("FileLoggerTest", level="INFO", log_file=dummy_log_file)
    logger_file.info("This is an info message for console and file.")
    logger_file.error("This is an error message for console and file.")

    # Verify content of dummy log file
    try:
        with open(dummy_log_file, 'r') as f:
            log_content = f.read()
            print(f"\n--- Content of {dummy_log_file} ---")
            print(log_content)
            assert "info message for console and file" in log_content
            assert "error message for console and file" in log_content
        print("Log file content verified.")
    except Exception as e:
        print(f"Error verifying log file: {e}")
    finally:
        # Clean up dummy log file
        import os
        if os.path.exists(dummy_log_file):
            os.remove(dummy_log_file)
            logger_console.info(f"Removed dummy log file: {dummy_log_file}")

    logger_already_configured = setup_logger("ConsoleLoggerTest", level="INFO")
    logger_already_configured.info("This message should appear once, showing handler duplication is prevented.")


    print("\n--- Logger Setup Utility Test Complete ---")
