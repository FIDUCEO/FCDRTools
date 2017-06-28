from distutils.core import setup

from setuptools import find_packages

from fiduceo.fcdr.writer.fcdr_writer import FCDRWriter

setup(name='fcdr_tools', version=FCDRWriter._version, description='FIDUCEO CDR/FCDR read and write utilities', author='Tom Block', author_email='tom.block@brockmann-consult.de',
      url='http://www.fiduceo.eu', packages=find_packages(), install_requires=['numpy', 'xarray', 'netcdf4'])
