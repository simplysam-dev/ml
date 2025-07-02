from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT="-e ." #We don't want hypen e. inside the requirements list 

# This function will open the requirement.txt file and read each line. It will remove any \n character and then include it inside 
# requirement variable in the form of list

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements if req.strip()]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

#This function will have all info about the project and the packages required for this project to work
setup(
    name='mlproject',
    version='0.0.1',
    author="Sam",
    author_email="sam.thedatabuddy@gmail.com",
    packages= find_packages(),
    install_requires=get_requirements("requirements.txt")
)