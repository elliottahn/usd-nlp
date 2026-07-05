from setuptools import find_packages, setup

setup(
    name="usd-nlp",
    version="0.1.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    author="Seok-Hyun Ahn",
    description="Universal Scene Description for Multilingual NLP Pipelines",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/elliottahn/usd-nlp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Text Processing :: Linguistic",
    ],
)
