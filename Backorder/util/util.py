from Backorder.exception import ApplicationException
from Backorder.contants import *
import os,sys
import yaml
import numpy as np
import pandas as pd


def read_yaml_file(file_path : str )->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ApplicationException(e,sys) from e