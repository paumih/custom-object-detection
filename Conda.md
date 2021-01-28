Conda vs Miniconda vs Anaconda

**conda** is an **environment manager** and a **package manager.**  

It means that conda makes possible to:  
  + Create an environment with any available version of Python with conda create -n myenv python=3.6 (this will create a new environment that has python 3.6 version + other package dependencies of python tools like pip, certifi, setuptools,wheel,vs2015_runtime, etc.)
  + Install and update packages into an existing conda environments: e.g. conda install scipy  

**conda is NOT a binary command**, is a **Python package**. Therefore, to make conda work, you have to create a Python environment (e.g. using python venv) and install package conda into it (e.g. using pip install conda). This is where Anaconda installer and Miniconda installer comes in.

**Miniconda installer** = Python + conda (i.e.installs a version of python [and its dependencies], creates a base python environment, and installs conda in it)  

**Anaconda installer** = Python + conda + meta package anaconda (i.e. installs a version of python [and its dependencies], creates a base python environment, and installs conda in it + installs other data science packages)  

**meta Python pkg anaconda = about 160 other Python packages for daily use in data science**  

Anaconda installer = install python (and its dependencies) + create a new base environment + pip install conda + conda install anaconda data science packages