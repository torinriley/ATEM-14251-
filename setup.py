from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="atem",
    version="1.2.5",
    author="Torin Etheridge",
    author_email="torinriley220@gmail.com",
    description="Adaptive task execution and machine learning package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CapitalRobotics/ATEM",
    packages=find_packages(include=["atem", "atem.*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy>=1.18.0",
        "colorama>=0.4.6"
    ],
)