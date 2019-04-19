<h1 align="center">CosmiQ Works Evaluation Tools</h1>
<p align="center">
<a href="http://www.cosmiqworks.org"><img src="http://www.cosmiqworks.org/wp-content/uploads/2016/02/cropped-CosmiQ-Works-Logo_R_RGB.png" width="350" alt="CosmiQ Works"></a>
<br>
<br>
<img align="center" src="https://img.shields.io/pypi/v/cw-eval.svg" alt="PyPI">
<img align="center" src="https://img.shields.io/conda/vn/conda-forge/cw-eval.svg" alt="conda-forge">
<img align="center" src="https://travis-ci.com/CosmiQ/cw-eval.svg?branch=master" alt="build">
<img align="center" src="https://readthedocs.org/projects/cw-eval/badge/" alt="docs">
<img align="center" src="https://img.shields.io/github/license/cosmiq/cw-eval.svg" alt="license">
<img align="center" src="https://img.shields.io/docker/build/cosmiqworks/cw-eval.svg" alt="docker">
<a href="https://codecov.io/gh/CosmiQ/cw-eval"><img align="center" src="https://codecov.io/gh/CosmiQ/cw-eval/branch/master/graph/badge.svg" /></a>
</p>

- [Installation Instructions](#installation-instructions)
- [API Documentation](https://cw-eval.readthedocs.io/)
- [Dependencies](#dependencies)
- [License](#license)
---
This package is purpose-built to support evaluation of computer vision models for geospatial imagery. The functionality contained here is used in evaluation of the SpaceNet Challenges.

## Installation Instructions
Several packages require binaries to be installed before pip installing the other packages.  Conda is a simple way to install everything and their dependencies:

#### Conda
```
conda install -c conda-forge cw-eval
```

#### pip

You may use `pip` to install this package; however, note that one of the dependencies, [rtree](https://github.com/Toblerity/rtree), can require pre-installation of [libspatialindex](https://libspatialindex.github.io/) binaries. This can all be done by installing rtree using conda:
```
conda install -c conda-forge rtree
```
or by following the instructions for libspatialindex install.

Once you have dependencies set up, install as usual using `pip`:
```
pip install cw-eval
```
For bleeding-edge versions (use at your own risk), `pip install` from the dev branch of this repository:
```
pip install --upgrade git+https://github.com/CosmiQ/cw-eval.git@dev
```

#### Docker

You may also use our Docker container:
```
docker pull cosmiqworks/cw-eval
```

## API Documentation
See the [readthedocs](https://cw-eval.readthedocs.io/) page.


## Evaluation Metric
The evaluation metric for this competition is an F1 score with the matching algorithm inspired by Algorithm 2 in the [ILSVRC paper applied to the detection of building footprints](https://arxiv.org/pdf/1409.0575v3.pdf). For each building there is a geospatially defined polygon label to represent the footprint of the building. A SpaceNet entry will generate polygons to represent proposed building footprints.  Each proposed building footprint is either a “true positive” or a “false positive”.

* The proposed footprint is a “true positive” if the proposal is the closest (measured by the IoU) proposal to a labeled polygon AND the IoU between the proposal and the label is about the prescribed threshold of 0.5.
* Otherwise, the proposed footprint is a “false positive”.

There is at most one “true positive” per labeled polygon.
The measure of proximity between labeled polygons and proposed polygons is the Jaccard similarity or the “Intersection over Union (IoU)”, defined as:

![alt text](https://github.com/SpaceNetChallenge/utilities/blob/master/content/IoU.jpg "IoU")

The value of IoU is between 0 and 1, where closer polygons have higher IoU values.

The F1 score is the harmonic mean of precision and recall, combining the accuracy in the precision measure and the completeness in the recall measure. For this competition, the number of true positives and false positives are aggregated over all of the test imagery and the F1 score is computed from the aggregated counts.

For example, suppose there are N polygon labels for building footprints that are considered ground truth and suppose there are M proposed polygons by an entry in the SpaceNet competition.  Let tp denote the number of true positives of the M proposed polygons.  The F1 score is calculated as follows:

![alt text](https://github.com/SpaceNetChallenge/utilities/blob/master/content/F1.jpg "IoU")

The F1 score is between 0 and 1, where larger numbers are better scores.

Hints:
* The images provided could contain anywhere from zero to multiple buildings.
* All proposed polygons should be legitimate (they should have an area, they should have points that at least make a triangle instead of a point or a line, etc).
* Use the [metric implementation code](https://github.com/SpaceNetChallenge/utilities/blob/master/python/evaluateScene.py) to self evaluate.
To run the metric you can use the following command:

```
spacenet_eval --help

spacenet_eval --proposal_csv ./TestCases_SpaceNet4/AOI_6_Atlanta_Test_v3_prop_1extra.csv \
              --truth_csv ./TestCases_SpaceNet4/AOI_6_Atlanta_Test_v3.csv \
              --challenge off-nadir \
              --output_file test.csv
```

## Dependencies
All dependencies can be found in the docker file [Dockerfile](./Dockerfile) or
[environment.yml](./environment.yml)

## License
See [LICENSE](./LICENSE.txt).

## Traffic
![GitHub](https://img.shields.io/github/downloads/cosmiq/cw-eval/total.svg)
![PyPI](https://img.shields.io/pypi/dm/cw-eval.svg)
![Conda](https://img.shields.io/conda/dn/conda-forge/cw-eval.svg)
