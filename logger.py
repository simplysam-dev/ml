import logging # module to log diagnostic message
import os # interact with file system, build paths and create directories
from datetime import datetime # return current date and time

LOG_FILE = f"{datetime.now().strftime('%m_%d,%Y_%H_%M_%S')}.log" # format for creating files

logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE) # C:/YourProject/logs/07_02,2025_17_36_45.log

os.makedirs(logs_path, exist_ok=True) #create log_path folder if not already existing

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE) # combining it together

logging.basicConfig(
    filename=LOG_FILE_PATH, # tells python where to save the log file
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s", # what should be the format of log message 
                                                                     #(timestamp, line number, name of logger, log leve (eg. ERROR), actual message)
    level=logging.INFO, #Only logs with severity level INFO or higher

)
'''
Testing Purpose

if __name__ == "__main__":  # ensures that this statement is only triggered when we run in directly, not when imported by other scripts
    logging.info("Logger setup successful")

'''