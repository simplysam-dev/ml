import sys #it provide access to functions and variables from python 
import logging

def error_message_detail(error, error_detail:sys): # Error message  (Exception), error detail calls for sys.exc_info()

    _,_,exc_tb=error_detail.exc_info()  #This catches the exception occurs in tuple format (Exception type, exception object, traceback object)
    # we ignore first two as we only need the traceback

    file_name = exc_tb.tb_frame.f_code.co_filename 
    # traceback object > current stack frame where the error happened > the filename of the code where the exception occurs

    error_message = "Error occured in python script name [{0}], line number [{1}], error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error) # display filename, line number, and represent error occured in str format
    )

    return error_message
    

class CustomException(Exception): # defining custom exception
    def __init__(self, error_message,error_detail:sys): 
        super().__init__(error_message) # calling parent class to store error message
        self.error_message= error_message_detail(error_message, error_detail=error_detail) # stores full detail of the error message

    def __str__(self):
        return self.error_message # ensures that the specific error message is generated 

''''  
Testing purpose: 
  
if __name__ == "__main__":  # ensures that this statement is only triggered when we run in directly, not when imported by other scripts
    try:
        1 / 0
    except Exception as e:
        logging.error("An exception occurred - Divide by Zero")
        raise CustomException(e, sys)
'''