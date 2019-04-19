<h1 align="center">Solaris</h1>
<h2 align="center">CosmiQ Works Geospatial Analysis Pipeline Toolkit
<p align="center">
<a href="http://www.cosmiqworks.org"><img src="http://www.cosmiqworks.org/wp-content/uploads/2016/02/cropped-CosmiQ-Works-Logo_R_RGB.png" width="350" alt="CosmiQ Works"></a>
<br>
<br>
<!-- <img align="center" src="https://img.shields.io/pypi/v/cw-eval.svg" alt="PyPI"> -->
<!-- <img align="center" src="https://img.shields.io/conda/vn/conda-forge/cw-eval.svg" alt="conda-forge">
<img align="center" src="https://travis-ci.com/CosmiQ/cw-eval.svg?branch=master" alt="build">
<img align="center" src="https://readthedocs.org/projects/cw-eval/badge/" alt="docs">
<img align="center" src="https://img.shields.io/github/license/cosmiq/cw-eval.svg" alt="license">
<img align="center" src="https://img.shields.io/docker/build/cosmiqworks/cw-eval.svg" alt="docker">
<a href="https://codecov.io/gh/CosmiQ/cw-eval"><img align="center" src="https://codecov.io/gh/CosmiQ/cw-eval/branch/master/graph/badge.svg" /></a>
</p> -->

- [Installation Instructions](#installation-instructions)
<!-- - [API Documentation](https://cw-eval.readthedocs.io/) -->
- [Dependencies](#dependencies)
- [License](#license)
---
This repository provides the source code for the CosmiQ Works `solaris` project, which provides software tools for:
- Tiling large-format overhead images and vector labels
- Converting between geospatial raster and vector formats and machine learning-compatible formats
- Performing semantic and instance segmentation, object detection, and related tasks using deep learning models designed specifically for overhead image analysis
- Evaluating performance of deep learning model predictions

## Installation Instructions
We recommend creating a `conda` environment with the dependencies defined in [environment.yml](./environment.yml) before installing `solaris`. After cloning the repository:
```
cd solaris
conda create -n solaris -f environment.yml
conda activate solaris
pip install .
```

#### pip

The package also exists on[ PyPI](https://pypi.org), but note that some of the dependencies, specifically [rtree](https://github.com/Toblerity/) and [gdal](https://www.gdal.org), are challenging to install without anaconda. We therefore recommend installing at least those dependency using `conda` before installing from PyPI.

```
conda install -c conda-forge rtree gdal
```
If you don't want to use `conda`, you can [install libspatialindex](https://libspatialindex.org), then `pip install rtree`. Installing GDAL without conda can be very difficult and approaches vary dramatically depending upon the build environment and version, but online resources may help with specific use cases.

Once you have that dependency set up, install as usual using `pip`:

```
pip install solaris
```

<!-- #### Docker

You may also use our Docker container:
```
docker pull cosmiqworks/solaris
``` -->

<!-- ## API Documentation
See the [readthedocs](https://cw-eval.readthedocs.io/) page. -->

## Dependencies
All dependencies can be found in the docker file [Dockerfile](./Dockerfile) or
[environment.yml](./environment.yml)

## License
See [LICENSE](./LICENSE.txt).
<!--
## Traffic
![GitHub](https://img.shields.io/github/downloads/cosmiq/cw-eval/total.svg)
![PyPI](https://img.shields.io/pypi/dm/cw-eval.svg)
![Conda](https://img.shields.io/conda/dn/conda-forge/cw-eval.svg) -->
