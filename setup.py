from setuptools import setup, find_packages


setup(
    name='karaboFAI',
    version="0.2.0",
    author='Jun Zhu',
    author_email='cas-support@xfel.eu',
    description='Azimuthal integration tool',
    long_description='Offline and online data analysis and visualization tool '
                     'for azimuthal integration of different data acquired '
                     'with various detectors at European XFEL.',
    url='',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'karaboFAI=karaboFAI.main_fai_gui:main_fai_gui',
            'karaboBDP=karaboFAI.main_bdp_gui:main_bdp_gui'
        ],
    },
    package_data={},
    install_requires=[
        'numpy>=1.14.5',
        'scipy>=1.1.0',
        'msgpack>=0.5.6',
        'msgpack-numpy>=0.4.4',
        'pyzmq>=17.1.2',
        'pyFAI>=0.15.0',
        'PyQt5>=5.11.0',
        'karabo-data>=0.2.0',
        'karabo-bridge>=0.2.0',
        'toolz',
        'silx>=0.8.0',
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
