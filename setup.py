import os
import re
from setuptools import setup, find_packages


def find_version():
    with open(os.path.join('karaboFAI', '__init__.py')) as fp:
        for line in fp:
            m = re.search(r'^__version__ = "(\d+\.\d+\.\d[a-z]*\d*)"', line, re.M)
            if m is not None:
                return m.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name='karaboFAI',
    version=find_version(),
    author='Jun Zhu',
    author_email='cas-support@xfel.eu',
    description='Fast analysis integration for 2D detectors',
    long_description='Real-time and off-line data analysis (azimuthal '
                     'integration, ROI, correlation, etc.) and visualization '
                     'tool for experiments using 2D detectors at '
                     'European XFEL.',
    url='',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'karaboFAI=karaboFAI.gui.main_gui:start',
            'karaboBDP=karaboFAI.gui.main_bdp_gui:main_bdp_gui'
        ],
    },
    package_data={
        'karaboFAI': [
            'gui/icons/*.png',
            'geometries/*.h5'
        ]
    },
    install_requires=[
        'numpy>=1.16.1',
        'scipy>=1.2.1',
        'msgpack>=0.5.6',
        'msgpack-numpy>=0.4.4',
        'pyzmq>=17.1.2',
        'pyFAI>=0.15.0',
        'PyQt5>=5.12.0',
        'karabo-data>=0.3.0',
        'karabo-bridge>=0.3.0',
        'toolz',
        'silx>=0.9.0',
        'cached-property>=1.5.1',
    ],
    extras_require={
        'docs': [
          'sphinx',
          'nbsphinx',
          'ipython',  # For nbsphinx syntax highlighting
        ],
        'test': [
          'pytest',
          'pytest-cov',
        ]
    },
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
