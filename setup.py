from setuptools import setup, find_packages

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
             'torch>=1.0.1',
             'affine>=2.2.1',
             'albumentations>=0.2.2',
             'rio-tiler>=1.2.4'
             ]

version = '0.0.1'

extra_reqs = {
    'test': ['mock', 'pytest', 'pytest-cov', 'codecov']}

setup(name='solaris',
      version=version,
      description="CosmiQ Works Geospatial Machine Learning Analysis Toolkit",
      classifiers=[
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
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
