import os
import sys
import subprocess
import logging
from setuptools import setup, find_packages
import re
import os


def get_version():
    VERSIONFILE = os.path.join('solaris', '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r'^__version__ = [\"\']*([\d.]+)[\"\']'
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger()


def check_output(cmd):
    # since subprocess.check_output doesn't exist in 2.6
    # we wrap it here.
    try:
        out = subprocess.check_output(cmd)
        return out.decode('utf')
    except AttributeError:
        # For some reasone check_output doesn't exist
        # So fall back on Popen
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out, err = p.communicate()
        return out


# check GDAL install
include_dirs = []
library_dirs = []
libraries = []
extra_link_args = []
gdal2plus = False
gdal_output = [None] * 4
gdalversion = None

try:
    gdal_version = subprocess.check_output(
        ['gdal-config', '--version']).decode('utf')
    gdal_config = os.environ.get('GDAL_CONFIG', 'gdal-config')

except Exception:
    sys.exit("GDAL must be installed to use `solaris`. See the documentation "
             "for more info. We recommend installing GDAL within a conda "
             "environment first, then installing solaris there.")




inst_reqs = ['shapely>=1.6.4',
             'fiona>=1.8.6',
             'pandas>=0.23.4',
             'geopandas>=0.4.0',
             'opencv-python==4.1.0.25',
             'numpy>=1.15.4',
             'tqdm>=4.28.1',
             'rtree>=0.8.3',
             'networkx>=2.2',
             'rasterio>=1.0.18',
             'scipy>=1.2.0',
             'scikit-image>=0.14.0',
             'tensorflow>=1.13.1',
             'torch>=1.1.0',
             'affine>=2.2.2',
             'albumentations>=0.2.3',
             'rio-tiler>=1.2.7',
             'pyyaml>=5.1',
             'torchvision>=0.3.0'
             ]


extra_reqs = {
    'test': ['mock', 'pytest', 'pytest-cov', 'codecov']}

project_name = 'solaris'
setup(name='solaris',
      version=get_version(),
      description="CosmiQ Works Geospatial Machine Learning Analysis Toolkit",
      classifiers=[
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: GIS'],
      author=u"CosmiQ Works",
      author_email='nweir@iqt.org',
      url='https://github.com/CosmiQ/solaris',
      license='Apache-2.0',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      zip_safe=False,
      include_package_data=True,
      install_requires=inst_reqs,
      extras_require=extra_reqs,
      entry_points={'console_scripts': [
          'geotransform_footprints = solaris.bin.geotransform_footprints:main',
          'make_graphs = solaris.bin.make_graphs:main',
          'make_masks = solaris.bin.make_masks:main',
          'mask_to_polygons = solaris.bin.mask_to_polygons:main',
          'spacenet_eval = solaris.bin.spacenet_eval:main'
          ]
      }
      )
