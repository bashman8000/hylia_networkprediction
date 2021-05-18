# Hylia: Studying Network Time-Series Data Made Easy

![](https://github.com/esnet/hylia_networkprediction/raw/master/static/hylia-logo.png)
---
![PyPI version](https://badge.fury.io/py/hylia.svg)
![Supported versions](https://img.shields.io/pypi/pyversions/hylia)



Hylia is a python library to study, process and forecast time series data produced in networking applications. The library contains a collection of multiple models from statistics (Arima, Holt-Winters) to complex deep learning models to train and inference forecasting challenges. 

## Documentation

- Example Tutorials
- API documentations

## Explanations
- Blog
- Video

## Install
Current installation of Hylia has been tested on Python 3.6. To get started  recommend to first setup a clean Python environment for your project with at least Python 3.6 using any of your favorite tool for instance, ([conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env"), [venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/) with or without [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)).

Once your environment is all set up you can install hylia using pip:

    pip install hylia

## Example 

Create a `TimeSeries` object from a Pandas DataFrame, and split it in train/validation series:

```python
import pandas as pd
from hylia import TimeSeries

df = pd.read_csv('network.csv', delimiter=",")
series = TimeSeries.from_dataframe(df, 'Month', '#bandwidth')
train, val = series.split_after(pd.Timestamp('1304241712'))
```

>The dataset used in this example can be downloaded from [here](https://github.com/esnet/hylia_networkprediction/blob/master/examples/networkdatasets/network.csv).

Fit an LSTM smoothing model, and make a prediction over the validation series' duration:

```python
from hylia.models import LSTM

model = LSTM()
model.fit(train)
prediction = model.predict(len(val))
```

Plot:
```python
import matplotlib.pyplot as plt

series.plot(label='actual')
prediction.plot(label='forecast', lw=2)
plt.legend()
plt.xlabel('Year')
```

<div style="text-align:center;">
<img src="https://github.com/esnet/hylia_networkprediction/blob/master/static/example.png" alt="hylia forecast example" />
</div>


Please feel free go over the example and tutorial notebooks in 
the [examples](https://github.com/esnet/hylia_networkprediction/tree/master/examples) directory.


## Datasets

Currently, hylia library supports the following network datasets: 

**Supported datasets:**

* NetFlow
* SNMP
* [PCAP](https://www.winpcap.org/)
* [sFLOW](https://sflow.org/about/index.php)

## Features

Currently, hylia library contains the following features: 

**Forecasting Models:**

* LSTM,
* SARIMA (For seasonality),
* Exponential smoothing,
* ARIMA,
* Facebook Prophet,
* FFT (Fast Fourier Transform),
* DDCRNN,



## Installation Guide

### Preconditions

A Conda environment is thus recommended because it will handle all of those in one go. The following steps assume running inside a conda environment. 

Create a conda environment for Python 3.7
(after installing [conda](https://docs.conda.io/en/latest/miniconda.html)):

    conda create --name <env-name> python=3.7

Activate your virtual environment

    conda activate <env-name>



### Install hylia

Install hylia with all available models: `pip install hylia`.


### Tests

A [gradle](https://gradle.org/) setup works best when used in a python environment, but the only requirement is to have `pip` installed for Python 3+

To run all tests at once just run
```bash
./gradlew test_all
```

alternatively you can run
```bash
./gradlew unitTest_all # to run only unittests
./gradlew coverageTest # to run coverage
./gradlew lint         # to run linter
```


### Documentation

To build documantation locally just run
```bash
./gradlew buildDocs
```
After that docs will be available in `./docs/build/html` directory. You can just open `./docs/build/html/index.html` using your favourite browser.


## Contact Us
See attached Licence to Lawrence Berkeley National Laboratory
Email: Mariam Kiran <mkiran@es.net>
