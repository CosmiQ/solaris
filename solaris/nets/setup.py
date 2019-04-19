from setuptools import setup, find_packages

version = '0.0.1'

# Runtime requirements.
inst_reqs = ["keras", "opencv-python", 'numpy', 'pyyaml', 'albumentations']  # TODO: UPDATE!

extra_reqs = {
    'test': ['mock', 'pytest', 'pytest-cov', 'codecov']}

setup(name='cw_nets',
      version=version,
      description=u"""Application-specific packaging of deep learning models for geospatial data""",
      classifiers=[
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: GIS'],
      keywords='spacenet machinelearning cosmiq deeplearning computervision',
      author=u"Nicholas Weir",
      author_email='nweir@iqt.org',
      url='https://github.com/CosmiQ/cw-nets',
      license='Apache-2.0',
      packages=find_packages(exclude=['tests', 'old']),
      zip_safe=False,
      include_package_data=True,
      install_requires=inst_reqs,
      extras_require=extra_reqs
      )
