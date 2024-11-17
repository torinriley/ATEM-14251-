from setuptools import setup, find_packages

setup(
    name="atem",
    version="1.2.0",
    author="Torin Etheridge",
    author_email="torinriley220@gmail.com",
    description="A Python package for adaptive task execution and machine learning integration.",
    url="https://github.com/CapitalRobotics/ATEM.git",
    packages=find_packages(include=["package", "package.*"]),
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
    ],
    entry_points={
        "console_scripts": [
            "atem-interpreter=atem_core.interpreter:main",
        ],
    },
)