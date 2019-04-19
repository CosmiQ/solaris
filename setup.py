from setuptools import setup, find_packages

version = '1.0.0'

readme = ''

# Runtime requirements.
inst_reqs = ["shapely", "rtree", "tqdm", "geopandas", "pandas"]

extra_reqs = {
    'test': ['mock', 'pytest', 'pytest-cov', 'codecov']}

setup(name='cw_eval',
      version=version,
      description=u"""Provide Evaluation Metrics for Machine Learning Challenges""",
      long_description=readme,
      classifiers=[
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: GIS'],
      keywords='spacenet machinelearning iou aws',
      author=u"David Lindenbaum and Nicholas Weir",
      author_email='nweir@iqt.org',
      url='https://github.com/CosmiQ/cw_eval',
      license='Apache-2.0',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      zip_safe=False,
      include_package_data=True,
      install_requires=inst_reqs,
      extras_require=extra_reqs,
      entry_points={
          'console_scripts': ['spacenet_eval=cw_eval.challenge_eval.spacenet_eval:main']
      }
      )
