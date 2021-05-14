# Hylia: Studying Network Time-Series Data Made Easy

<p align="center">
<img src="https://github.com/esnet/hylia_networkprediction/blob/master/static:images/hylia-logo.png" width="100%" height="100%" title="hylia-logo">
<p>
 
---
![Supported versions](https://img.shields.io/badge/python-3.7+-blue.svg)
![Supported versions](https://img.shields.io/badge/python-3.7+-blue.svg)
![Supported versions](https://img.shields.io/badge/python-3.7+-blue.svg)


Hylia is a python libary to study, process and forecast time series data produced in networking applications. The library contains a collection of multiple models from statistics (Arima, Holt-Winters) to complex deep learning models to train and inference forecasting challenges. 

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
train, val = series.split_after(pd.Timestamp('20210000'))
```

>The dataset used in this example can be downloaded from here.

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
<img src="https://github.com/esnet/hylia_networkprediction/blob/master/static:images/example.png" alt="hylia forecast example" />
</div>


Please feel free go over the example and tutorial notebooks in 
the [examples](https://github.com/esnet/hylia_networkprediction/tree/master/examples) directory.

## Contact Us
See attached Licence to Lawrence Berkeley National Laboratory
Email: Mariam Kiran <mkiran@es.net>
