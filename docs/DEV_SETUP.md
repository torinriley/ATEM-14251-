# ATEM Development Environment Setup

This document provides a comprehensive guide to setting up the development environment for working on the ATEM project.

---

## Prerequisites

Before getting started, ensure the following tools and software are installed on your system:

1. **Python**  
   - Version: `>=3.7`
   - [Download Python](https://www.python.org/downloads/)

2. **pip**  
   - Python's package manager should be installed with Python.

3. **Git**  
   - [Download Git](https://git-scm.com/downloads)

4. **Virtual Environment (optional but recommended)**  
   - To isolate dependencies: `python -m venv .venv`

5. **IDE/Editor**  
   - Recommended: [PyCharm](https://www.jetbrains.com/pycharm/) or [VS Code](https://code.visualstudio.com/)

6. **Optional Tools**  
   - Docker: If you prefer containerized development.
   - Jupyter Notebook: For testing and experimentation.

---

## Project Setup

### 1. Clone the Repository

Clone the ATEM repository to your local machine:

```bash
git clone https://github.com/CapitalRobotics/ATEM.git
cd ATEM
```

### 2. Set Up a Virtual Environment (Recommended)

Create a virtual environment to isolate the dependencies:

```bash
python -m venv .venv
```


### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Running the Tests
Run unit tests to ensure everything is set up correctly:

```bash
pytest
```
