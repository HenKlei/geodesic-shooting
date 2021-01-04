# pyLDDMM
### A documented implementation of Large deformation diffeomorphic metric mapping (LDDMM) in python.

The LDDMM algorithm, introduced in Faisal Beg et al. (2005), has been a
centerpiece of establishing diffeomorphic registration. It expands the large
deformation model based on the model of viscous fluids (Christensen, Rabbitt, and Miller 1996) by proposing a gradient-descent based optimization
algorithm based on the derivation of the Euler-Lagrange equation for the
variational minimization of vector fields.

While many open source implementations of the algorithm exist, most of them are either implemented in hard to understand languages (most notable C++, as in [this](https://github.com/frankyeh/TIPL/blob/master/reg/lddmm.hpp) example), or not commented at all (such as [this](https://github.com/Ryo-Ito/image_registration) python implementation).

This implementation aims at making the algorithm accessible for learning. It stays true to the original paper ["Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms" by Faisal Beg et al. (2005)](https://link.springer.com/article/10.1023/B:VISI.0000043755.93987.aa), and avoids further complications. The [code](pyLDDMM/LDDMM.py) contains explanetory comments, frequently references steps and equations of the paper, and keeps variable names consistent with the paper.

This implementation is not focused at performance, or even at obtaining best registration results. While small speedups can be achieved even with minor changes to the algorithm, this implementation closely follows the LDDMM algorithm as explained in the paper. In an effort to make the code easily understandable, only the registration of 1d and 2d images is supported. Volumentric images are not supported. 

## Installation
The provided software runs on python3. All required dependencies can easily be instaled with pip:
```
pip3 install -r requirements.txt
```
The program itself can be installed by running
```
pip3 install .
```

## Examples
Three examples are provided. A one-dimensional example, a circle translation, and morphing a circle to a square.

The examples can be executed in the examples folder with:
```
python3 1d_example.py
python3 translation_example.py
python3 circle_to_square_example.py
```
Output files are written into the directory `examples/example_images/`.

## Documentation
The commented code is in the file [pyLDDMM/LDDMM.py](pyLDDMM/LDDMM.py). 
