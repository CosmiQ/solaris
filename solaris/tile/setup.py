from setuptools import setup, find_packages

version = '0.2.0'

with open('README.md') as f:
    readme = f.read()

# Runtime requirements.
inst_reqs = ["rio-tiler", "shapely", "geopandas"]

extra_reqs = {
    'test': ['mock', 'pytest', 'pytest-cov', 'codecov']}

setup(name='cw_tiler',
      version=version,
      description=u"""Get UTM tiles for SpaceNet Dataset or arbitrary GeoTiffs""",
      long_description=readme,
      classifiers=[
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: GIS'],
      keywords='raster aws tiler gdal rasterio spacenet machinelearning',
      author=u"David Lindenbaum and Nick Weir",
      author_email='dlindenbaum@iqt.org, nweir@iqt.org',
      url='https://github.com/CosmiQ/cw-tiler',
      license='BSD',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      zip_safe=False,
      install_requires=inst_reqs,
      extras_require=extra_reqs)
