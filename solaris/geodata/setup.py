from setuptools import setup, find_packages
version = '0.1.0'

# Runtime requirements.
inst_reqs = ["shapely==1.6.4",
             "rtree==0.8.3",
             "geopandas==0.4.0",
             "pandas==0.23.4",
             "networkx==2.2",
             "rasterio==1.0.18",
             "tqdm==4.28.1",
             "numpy==1.15.4",
             "scipy==1.2.0",
             "scikit-image==0.14",
             "affine==2.2.1"]

extra_reqs = {
    'test': ['mock', 'pytest', 'pytest-cov', 'codecov']}

console_scripts = [
    'geotransform_footprints=cw_geodata.bin.geotransform_footprints:main',
    'make_masks=cw_geodata.bin.make_masks:main',
    'make_graphs=cw_geodata.bin.make_graphs:main'
    ]

setup(name='cw_geodata',
      version=version,
      description=u"""Geospatial raster and vector data processing for ML""",
      classifiers=[
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: GIS'],
      keywords='spacenet machinelearning gis geojson',
      author=u"Nicholas Weir",
      author_email='nweir@iqt.org',
      url='https://github.com/CosmiQ/cw-geodata',
      license='Apache-2.0',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      zip_safe=False,
      include_package_data=True,
      install_requires=inst_reqs,
      extras_require=extra_reqs,
      entry_points={'console_scripts': console_scripts}
      )
