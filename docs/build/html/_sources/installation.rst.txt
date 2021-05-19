Installation
===================================

Installing the latest release
*****************************

Installing Hylia is the first step towards building your first machine learning model in Hylia. Installation is easy and takes only a few minutes. All hard dependencies are also installed with Hylia. `Click here <https://github.com/unit8co/darts/tree/master/requirements>`_ to see the complete list of hard dependencies. 

In order to avoid potential conflicts with other packages, it is strongly recommended to use a virtual environment, e.g. python3 virtualenv (see `python3 virtualenv documentation <https://docs.python.org/3/tutorial/venv.html>`_) or `conda environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_. Using an isolated environment makes it possible to install a specific version of hylia and its dependencies independently of any previously installed Python packages. See an example below of how to create a conda environment and install Hylia. 

.. code-block:: python

    # create a conda environment
    conda create --name yourenvname python=3.6

    # activate conda environment
    conda activate yourenvname

    # install hylia
    pip install hylia

    # create notebook kernel connected with the conda environment
    python -m ipykernel install --user --name yourenvname --display-name "display-name"


Installing the full version 
***************************
Hylia's default installation is a slim version of hylia which only installs hard dependencies that are `listed here <https://github.com/unit8co/darts/tree/master/requirements>`_. To install a specific version of hylia, use the following command:

.. code-block:: python

    # install the full version of hylia
    pip install hylia==0.1.1


Hylia on GPU
***************
Coming soon

Recommended environment for use
*******************************

You can use Hylia in your choice of Integrated Development Environment (IDE) but since it uses html and several other interactive widgets, it is optimized for use within a notebook environment, be it `Jupyter Notebook <https://jupyter.org/>`_, `Jupyter Lab <https://jupyterlab.readthedocs.io/en/stable/>`_, `Azure Notebooks <https://notebooks.azure.com/>`_ or `Google Colab <https://colab.research.google.com/>`_.

- `Learn how to install Jupyter Notebook <https://jupyter.readthedocs.io/en/latest/install.html>`_
- `Learn how to install Jupyter Lab <https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html>`_
- `Get Started with Azure Notebooks <https://notebooks.azure.com/>`_
- `Get Started with Google Colab <https://colab.research.google.com/>`_
- `Get Started with Anaconda Distribution <https://www.anaconda.com/>`_

