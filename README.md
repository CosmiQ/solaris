<p align="center">
<img src="https://github.com/CosmiQ/solaris/raw/main/static/sol_logo.png" width="350" alt="solaris">
</p>
<h2 align="center">An open source ML toolkit for overhead imagery.</h2>
<p align="center">
<img align="center" src="https://img.shields.io/pypi/pyversions/solaris.svg" alt="PyPI python version" href="https://pypi.org/project/solaris/">
<img align="center" src="https://img.shields.io/pypi/v/solaris.svg" alt="PyPI" href="https://pypi.org/project/solaris/">
<!-- <img align="center" src="https://img.shields.io/conda/vn/conda-forge/cw-eval.svg" alt="conda-forge"> -->
<img align="center" src="https://readthedocs.org/projects/solaris/badge/" alt="docs">
<img align="center" src="https://img.shields.io/github/license/CosmiQ/solaris.svg" alt="license">
<a href="https://codecov.io/gh/CosmiQ/solaris"><img align="center" src="https://codecov.io/gh/CosmiQ/solaris/branch/main/graph/badge.svg" /></a>
</p>

## This is a beta version of solaris which may continue to develop. Please report any bugs through issues!

- [This is a beta version of solaris which may continue to develop. Please report any bugs through issues!](#this-is-a-beta-version-of-solaris-which-may-continue-to-develop-please-report-any-bugs-through-issues)
- [- License](#--license)
- [Documentation](#documentation)
- [Installation Instructions](#installation-instructions)
    - [pip](#pip)
- [Dependencies](#dependencies)
- [License](#license)
---

Solaris is an open source GeoAI ML toolkit for preprocessing and post processing workflows that are common to machine learning with geospatial imagery. It handles common functionality for preprocessing geotiffs and vector formats into formats that deep learning frameworks can interpret. It also handles post-processing and evaluating detection results, making it easier to compare different models. See the [ROADMAP.md](ROADMAP.md) for a description of how v0.5 differs from v0.4 of solaris.

This repository provides the source code for the `solaris` project, which provides software tools for:
- Tiling large-format overhead images and vector labels
- Converting between geospatial raster and vector formats and machine learning-compatible formats
- Evaluating performance of deep learning model predictions, including semantic and instance segmentation, object detection, and related tasks

## Documentation
The full documentation for `solaris` can be found at https://solaris.readthedocs.io, and includes:
- A summary of `solaris`
- Installation instructions
- API Documentation
- Tutorials for common uses

The documentation is still being improved, so if a tutorial you need isn't there yet, check back soon or post an issue!

## Installation Instructions

_coming soon_: One-command installation from conda-forge.

We recommend creating a `conda` environment with the dependencies defined in [environment.yml](./environment.yml) before installing `solaris`. After cloning the repository:
```
cd solaris
```

If you're installing on a system with GPU access:
```
conda env create -n solaris -f environment-gpu.yml
```
Otherwise:
```
conda env create -n solaris -f environment.yml
```

Finally, regardless of your installation environment:
```
conda activate solaris
pip install .
```

#### pip


The package also exists on[ PyPI](https://pypi.org), but note that some of the dependencies, specifically [rtree](https://github.com/Toblerity/rtree) and [gdal](https://www.gdal.org), are challenging to install without anaconda. We therefore recommend installing at least those dependencies using `conda` before installing from PyPI.

```
conda install -c conda-forge rtree gdal=2.4.1
pip install solaris
```

If you don't want to use `conda`, you can [install libspatialindex](https://libspatialindex.org), then `pip install rtree`. Installing GDAL without conda can be very difficult and approaches vary dramatically depending upon the build environment and version, but [the rasterio install documentation](https://rasterio.readthedocs.io/en/stable/installation.html) provides OS-specific install instructions. Simply follow their install instructions, replacing `pip install rasterio` with `pip install solaris` at the end.

<!-- #### Docker

You may also use this Docker container:
```
docker pull CosmiQ/solaris
``` -->

<!-- ## API Documentation
See the [readthedocs](https://cw-eval.readthedocs.io/) page. -->

## Dependencies
All dependencies can be found in the requirements file [./requirements.txt](requirements.txt) or
[environment.yml](./environment.yml)

## License
See [LICENSE](./LICENSE.txt).
<!--
## Traffic
![GitHub](https://img.shields.io/github/downloads/CosmiQ/cw-eval/total.svg)
![PyPI](https://img.shields.io/pypi/dm/cw-eval.svg)
![Conda](https://img.shields.io/conda/dn/conda-forge/cw-eval.svg) -->
