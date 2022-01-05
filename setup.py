import os
import sys
import subprocess
import logging
from setuptools import setup, find_packages
import re


def get_version():
    VERSIONFILE = os.path.join('solaris', '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r'^__version__ = [\"\']*([\d\w.]+)[\"\']'
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


on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    inst_reqs = ['sphinx_bootstrap_theme']
else:
    inst_reqs = ['pip>=19.0.3',
                 'affine>=2.3.0',
                 'albumentations==0.4.3',
                 'fiona>=1.7.13',
                 'gdal>=3.0.2',
                 'geopandas>=0.7.0',
                 'matplotlib>=3.1.2',
                 'networkx>=2.4',
                 'numpy>=1.17.3',
                 'opencv-python>=4.1',
                 'pandas>=0.25.3',
                 'pyproj>=2.1',
                 'torch>=1.3.1',
                 'pyyaml==5.2',
                 'rasterio>=1.0.23',
                 'requests==2.22.0',
                 'rtree>=0.9.3',
                 'scikit-image>=0.16.2',
                 'scipy>=1.3.2',
                 'shapely>=1.7.1dev',
                 'torchvision>=0.5.0',
                 'tqdm>=4.40.0',
                 'urllib3>=1.25.7',
                 'tensorflow==1.13.1'
                 ]


extra_reqs = {
    'test': ['mock', 'pytest', 'pytest-cov', 'codecov']}

# workaround until new shapely release is out
os.system('pip install  git+git://github.com/toblerity/shapely@master')


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
      dependency_links=['https://github.com/toblerity/shapely/tarball/master#egg=shapely-1.7.1dev'],
      entry_points={'console_scripts': [
          'geotransform_footprints = solaris.bin.geotransform_footprints:main',
          'make_graphs = solaris.bin.make_graphs:main',
          'make_masks = solaris.bin.make_masks:main',
          'mask_to_polygons = solaris.bin.mask_to_polygons:main',
          'spacenet_eval = solaris.bin.spacenet_eval:main',
          'solaris_run_ml = solaris.bin.solaris_run_ml:main'
          ]
      }
      )
