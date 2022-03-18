from setuptools import setup, find_packages


dependencies = [
    'numpy',
    'scipy',
    'scikit-image',
    'matplotlib',
]

setup(
    name='geodesic-shooting',
    version='0.1.0',
    description='Python implementation of the geodesic shooting algorithm',
    author='Hendrik Kleikamp',
    maintainer='Hendrik Kleikamp',
    maintainer_email='hendrik.kleikamp@uni-muenster.de',
    packages=find_packages(),
    install_requires=dependencies,
)
