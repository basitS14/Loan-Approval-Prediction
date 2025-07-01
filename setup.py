from typing import List
from setuptools import find_packages , setup

HYPHEN_DOT = '-e .'

def get_requirements(file_path:str) -> List[str] :

    requirements = []

    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n" , "") for req in requirements]

        if HYPHEN_DOT in requirements:
            requirements.remove(HYPHEN_DOT)
    
    return requirements

setup(
    author="basit",
    version="0.1",
    name="Loan Approval",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)