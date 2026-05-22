"""Setup script for the USD-NLP toolkit."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="usd-nlp",
    version="0.1.0",
    author="Seok-Hyun Ahn",
    author_email="elliott.ahn@gist.ac.kr",
    description=("Universal Scene Description composition model "
                 "for multilingual NLP pipelines"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elliottahn/usd-nlp",
    project_urls={
        "Source": "https://github.com/elliottahn/usd-nlp",
        "Bug Tracker": "https://github.com/elliottahn/usd-nlp/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=[],  # zero external dependencies
)
