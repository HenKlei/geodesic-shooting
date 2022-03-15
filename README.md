# Image registration using geodesic shooting and landmark matching

The field of image registration has its origins and main applications in the analysis of medical
image data. Given two images of the same object, the goal is to find a transformation such that
the transformed first image matches the second image.

The algorithms implemented in this package use diffeomorphic transformations of the underlying
domain to map the two images onto each other.

## Installation
To install the software, you first have to clone the repository by running
```
git clone https://github.com/HenKlei/geodesic-shooting.git
```
or (in case you prefer using `ssh`)
```
git clone git@github.com:HenKlei/geodesic-shooting.git
```

To install the package, switch to the folder `geodesic-shooting` and use
```
pip install .
```
This will install the required dependencies for the basic functionality and afterwards install the
package itself. If you want to make use of the full functionality (including visualizations,
functions for input and output of images, etc.), you can install the optional requirements via
```
pip install -r requirements-optional.txt
```
It is recommended to use a virtual environment to install the package in.

## Examples
The package comes with several examples for all the algorithms implemented. The examples are
collected in the `examples` folder and sorted by registration algorithm. For instance, one- and
two-dimensional examples with plain numpy arrays, a circle translation, and morphing a circle to a
square are included.

## Documentation
The commented code for the two main algorithms can be found in the following files:
- Geodesic shooting: [geodesic_shooting.py](geodesic_shooting/geodesic_shooting.py)
- Landmark matching: [landmark_shooting.py](geodesic_shooting/landmark_shooting.py)
