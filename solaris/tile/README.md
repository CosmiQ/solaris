<h1 align="center">CosmiQ Works Dataset Tiling Tools</h1>
<p align="center">
<a href="http://www.cosmiqworks.org"><img src="http://www.cosmiqworks.org/wp-content/uploads/2016/02/cropped-CosmiQ-Works-Logo_R_RGB.png" width="350" alt="CosmiQ Works"></a>
<br>
<br>
<img align="center" src="https://img.shields.io/pypi/pyversions/cw-tiler.svg">
<img align="center" src="https://img.shields.io/conda/vn/conda-forge/cw-tiler.svg">
<img align="center" src="https://readthedocs.org/projects/cw-tiler/badge/" alt="docs">
<img align="center" src="https://img.shields.io/github/license/cosmiq/cw-tiler.svg" alt="license">
</p>

- [Installation Instructions](#installation-instructions)
- [API Documentation](https://cw-eval.readthedocs.io/)
- [Dependencies](#dependencies)
- [License](#license)
- [Download Instructions](#spacenet-dataset-download-instructions)
---
Modeled off of capabilities from [rio-tiler](https://github.com/mapbox/rio-tiler) and designed for use with SpaceNet datasets, this library provides code for tiling geospatial imagery datasets into manageable pieces.


## Installation Instructions
Several packages require binaries to be installed before pip installing the other packages.  Conda is a simple way to install everything and their dependencies:

#### Conda
_note: This functionality is not ready as of 12.11.18. Check back soon._
```
conda install -c conda-forge cw-tiler
```

#### pip

You may use `pip` to install this package; however, note that one of the dependencies, [rtree](https://github.com/Toblerity/rtree), can require pre-installation of [libspatialindex](https://libspatialindex.github.io/) binaries. This can all be done by installing rtree using conda:
```
conda install -c conda-forge rtree
```
or by [following the instructions for libspatialindex installation](https://libspatialindex.github.io/).

Once you have dependencies set up, install as usual using `pip`:
```
pip install cw-tiler
```
For bleeding-edge versions (use at your own risk), `pip install` from the dev branch of this repository:
```
pip install --upgrade git+https://github.com/CosmiQ/cw-tiler.git@dev
```
## API Documentation
See the [readthedocs](https://cw-tiler.readthedocs.io/) page.

## Dependencies
All dependencies can be found in the docker file [Dockerfile](./Dockerfile) or
[environment.yml](./environment.yml)

## License
See [LICENSE](./LICENSE.txt).

## SpaceNet Dataset Download Instructions
Further download instructions for the [SpaceNet Dataset](https://github.com/SpaceNetChallenge/utilities/tree/master/content/download_instructions) can be found [here](https://github.com/SpaceNetChallenge/utilities/tree/master/content/download_instructions).
