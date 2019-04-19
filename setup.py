from setuptools import setup, find_packages
from pip.req import parse_requirements


version = '0.0.1'

requirements_path = 'requirements.txt'
# Runtime requirements.
inst_reqs = parse_requirements(requirements_path)

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
