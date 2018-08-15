#!/usr/bin/env python
from os.path import dirname
from setuptools import setup, find_packages


def _get_version_string():
    try:
        from karabo.packaging.versioning import get_package_version
    except ImportError:
        # print("WARNING: Karabo framework not found! Version will be blank!")
        return "0.1.0"

    return get_package_version(dirname(__file__))


setup(
    name='fxeAzimuthalIntegration',
    version=_get_version_string(),
    author='Jun Zhu',
    author_email='jun.zhu@xfel.eu',
    description='LPD azimuthal integration',
    long_description='Online and offline tool for LPD azimuthal integration '
                     'at FXE instrument, EuXFEL',
    url='',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    entry_points={
        'console_scripts': [
            'fxe-gui=fxeAzimuthalIntegration.gui:fxe_ai',
        ],
    },
    package_data={},
    requires=[],
)
