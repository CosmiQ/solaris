from setuptools import setup, find_packages

version = '0.0.1'

# Runtime requirements.
inst_reqs = ["shapely", "rtree", "tqdm", "geopandas", "pandas"]

extra_reqs = {
    'test': ['mock', 'pytest', 'pytest-cov', 'codecov']}

setup(name='cw_eval',
      version=version,
      description="CosmiQ Works Geospatial Analysis Pipeline Toolkit",
      classifiers=[
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: GIS'],
      keywords='spacenet machinelearning iou aws',
      author=u"CosmiQ Works",
      author_email='nweir@iqt.org',
      url='https://github.com/CosmiQ/solaris',
      license='Apache-2.0',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      zip_safe=False,
      include_package_data=True,
      install_requires=inst_reqs,
      extras_require=extra_reqs,
      entry_points={
      }
      )
