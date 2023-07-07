from setuptools import setup, find_packages

setup(
    name="DL Package",
    version="0.1",
    author="Dazhong Li",
    description="Personal Package",
    packages=find_packages("src"),
    package_dir={"": "src"}
)

