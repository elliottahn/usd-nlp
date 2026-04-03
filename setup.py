from setuptools import setup, find_packages
setup(
    name="usd-nlp",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    author="Seok-Hyun Ahn",
    description="Universal Scene Description for Multilingual NLP Pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/[REPO]/usd-nlp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Text Processing :: Linguistic",
    ],
)
