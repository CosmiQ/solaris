<h1 align="center">CosmiQ Works Geospatial Data Processing Tools for ML</h1>
<p align="center">
<a href="http://www.cosmiqworks.org"><img src="http://www.cosmiqworks.org/wp-content/uploads/2016/02/cropped-CosmiQ-Works-Logo_R_RGB.png" width="350" alt="CosmiQ Works"></a>
<br>
<br>
<br>
<img align="center" src="https://img.shields.io/pypi/v/cw-geodata.svg" alt="PyPI">
<!-- <img align="center" src="https://img.shields.io/conda/vn/conda-forge/cw-eval.svg" alt="conda-forge"> -->
<img align="center" src="https://travis-ci.com/CosmiQ/cw-geodata.svg?branch=master" alt="build">
<img align="center" src="https://readthedocs.org/projects/cw-geodata/badge/" alt="docs">
<img align="center" src="https://img.shields.io/github/license/cosmiq/cw-geodata.svg" alt="license">
<!-- <img align="center" src="https://img.shields.io/docker/build/cosmiqworks/cw-eval.svg" alt="docker"> -->
<a href="https://codecov.io/gh/CosmiQ/cw-geodata"><img align="center" src="https://codecov.io/gh/CosmiQ/cw-geodata/branch/master/graph/badge.svg" /></a>
</p>

__This package is currently under active development. Check back soon for a mature version.__

- [Installation Instructions](#installation-instructions)
- [API Documentation](https://cw-geodata.readthedocs.io/)
- [Dependencies](#dependencies)
- [License](#license)
---
This package is built to:
- Enable management and interconversion of geospatial data files without requiring understanding of coordinate reference systems, geospatial transforms, etc.
- Enable creation of training targets for segmentation and object detection from geospatial vector data (_i.e._ geojsons of labels) without requiring understanding of ML training target formats.

## Installation Instructions

We recommend creating a `conda` environment with the dependencies defined in [environment.yml](https://github.com/CosmiQ/cw-geodata/blob/master/environment.yml) before installing `cw-geodata`:

```
git clone https://github.com/cosmiq/cw-geodata.git
cd cw-geodata
conda create -n cw-geodata -f environment.yml
conda activate cw-geodata
pip install .
```

The package also exists on[ PyPI](https://pypi.org), but note that some of the dependencies, specifically [rtree](https://github.com/Toblerity/), is challenging to install without anaconda. We therefore recommend installing at least that dependency using `conda` before installing from PyPI.
```
conda install -c conda-forge rtree=0.8.3
```

If you don't want to use `conda`, you can [install libspatialindex](https://libspatialindex.org), then `pip install rtree`.

Once you have that dependency set up, install as usual using `pip`:

```
pip install cw-geodata
```

For bleeding-edge versions (use at your own risk), `pip install` from the dev branch of this repository:

```
pip install --upgrade git+https://github.com/CosmiQ/cw-geodata.git@dev
```

## API Documentation
API documentation can be found [here](https://cw-geodata.readthedocs.io)

## Dependencies
All dependencies can be found in [environment.yml](./environment.yml)

## License
See [LICENSE](./LICENSE.txt).
