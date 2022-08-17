from setuptools import PackageFinder, find_packages, setup
from typing import List

# Declaring Variables for setup function

PROJECT_NAME = "Backorder Prediction"
VERSION = "0.0.1ins"
AUTHOR = "Shivansh Kaushal and Chirag Sharma"
DESCRIPTION = "This application helps to predict products which are in backorder"
REQUIREMENT_FILE_NAME = "requirements.txt"

def get_requirements_list()->List[str]:
    """
    Description: This function is going to return list of requirement mentioned in requirements.txt
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        return requirement_file.readlines().remove("-e .")

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires = get_requirements_list()
)

if __name__ == "__main__":
    print(get_requirements_list())