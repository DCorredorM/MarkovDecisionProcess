import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mdpy", # Replace with your own username
    version="0.0.1",
    author="David Corredor M",
    author_email="d.corredor@uniandes.edu.co",
    description="A package for modeling and solving MDPs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DCorredorM/MarkovDecisionProcess",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)