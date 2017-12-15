'''
Setup script for MaskAccessor
author: Leevi Annala
'''
from setuptools import setup, find_packages

with open('LICENSE') as f:
    LICENSE_FILE = f.read()

INSTALL_REQUIRES = ['xarray',
                    'numpy']


setup(name='maskacc',
      version='0.1.0',
      description='',
      author='Leevi Annala',
      author_email='lealanna@student.jyu.fi',
      url='',
      install_requires=INSTALL_REQUIRES,
      packages=find_packages(),
      license=LICENSE_FILE
     )
