from pathlib import Path
from setuptools import setup, find_packages


# def read_requirements(path):
#     return list(Path(path).read_text().splitlines())


# base_reqs = read_requirements('requirements/core.txt')
# pmdarima_reqs = read_requirements('requirements/pmdarima.txt')
# torch_reqs = read_requirements('requirements/torch.txt')
# fbprophet_reqs = read_requirements('requirements/fbprophet.txt')

# all_reqs = base_reqs + pmdarima_reqs + torch_reqs + fbprophet_reqs

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()


URL = 'https://github.com/esnet/hylia_networkprediction'


PROJECT_URLS = {
#    'Bug Tracker': 'https://github.com/esnet/hylia_networkprediction',
    'Documentation': URL,
    'Source Code': 'https://github.com/esnet/hylia_networkprediction'
}


setup(
      name='hylia',
      version="0.1.0",
      description='A python time series library for network operations.',
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      project_urls=PROJECT_URLS,
      url=URL,
      maintainer='Daphne',
      maintainer_email='mkiran@es.net',
      license='Lawrence Berkeley National Laboratory',
      packages=find_packages(),
      #packages=['hylia'],
      install_requires=['pmdarima',
                        'numpy>=1.19.5',
                        'torch>=1.8.1',
#                        'fbprophet>=0.7.1',
                        'scipy>=1.6.2',
                        'statsmodels',
                        'pandas>=1.2.3',
                        'ipython>=7.22.0',                     
                        'tqdm>=4.60.0',
                        'holidays>=0.11.1',
                        'scikit-learn>=0.24.1',
                        'pystan>=2.19.1.1',
                        'tensorboard>=2.4.1',
                       ],
      #install_requires=all_reqs,
      package_data={
          'hylia': ['py.typed'],
      },
      zip_safe=False,
      python_requires='>=3.6',
      classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            ('Programming Language :: Python :: '
             'Implementation :: PyPy')
      ],
      keywords='network time series prediction library'
)
