from distutils.log import error
import os
import sys

class ApplicationException(Exception):
    
    def __init__(self, error_message:Exception, error_details:sys):
        super().__init__(error_message)
        self.error_message = ApplicationException.get_detailed_error_message(error_message=error_message,
                                                                                error_details=error_details)

    @staticmethod
    def get_detailed_error_message(error_message:Exception, error_details:sys)->str:
        """
        error_message: Exception object
        error_details: object of sys module
        """

        _, _, exec_tb = error_details.exc_info()

        line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename

        error_message = f"""
        Error occured in script: [{file_name}] at 
        line number: [{line_number}] 
        error message: [{error_message}]"""
        return error_message

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return ApplicationException.__name__.str()