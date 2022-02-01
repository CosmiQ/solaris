import logging
import os
import re
import subprocess
import sys

from setuptools import find_packages, setup


def get_version():
    VERSIONFILE = os.path.join("solaris", "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = [\"\']*([\d\w.]+)[\"\']"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger()


def check_output(cmd):
    # since subprocess.check_output doesn't exist in 2.6
    # we wrap it here.
    try:
        out = subprocess.check_output(cmd)
        return out.decode("utf")
    except AttributeError:
        # For some reasone check_output doesn't exist
        # So fall back on Popen
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out, err = p.communicate()
        return out


on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    inst_reqs = ["sphinx_bootstrap_theme"]
else:
    inst_reqs = [
        "pip>=19.0.3",
        "affine>=2.3.0",
        "fiona>=1.7.13",
        "geopandas>=0.7.0",
        "matplotlib>=3.1.2",
        "numpy>=1.17.3",
        "opencv-python>=4.1",
        "pandas>=0.25.3",
        "pyproj>=2.1",
        "PyYAML>=5.4",
        "rasterio>=1.0.23",
        "rio-cogeo>=3.0.2",
        "requests==2.22.0",
        "rtree>=0.9.3",
        "scikit-image>=0.16.2",
        "scipy>=1.3.2",
        "shapely>=1.7.1dev",
        "tqdm>=4.40.0",
        "urllib3<1.26",
    ]


extra_reqs = {"test": ["mock", "pytest", "pytest-cov", "codecov"]}


project_name = "solaris"
setup(
    name="solaris",
    version=get_version(),
    description="Geospatial Machine Learning Preprocessing and Evaluation Toolkit",
    classifiers=[
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    author="Ryan Avery",
    author_email="ryan@developmentseed.org",
    url="https://github.com/CosmiQ/solaris",
    license="MIT",
    packages=find_packages(exclude=["ez_setup", "examples", "tests"]),
    zip_safe=False,
    include_package_data=True,
    install_requires=inst_reqs,
    extras_require=extra_reqs,
    entry_points={
        "console_scripts": [
            "geotransform_footprints = solaris.bin.geotransform_footprints:main",
            "make_masks = solaris.bin.make_masks:main",
        ]
    },
)
