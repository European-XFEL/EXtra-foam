from setuptools import setup, find_packages


REQUIREMENTS = open('requirements.txt', encoding='utf-8').readlines()
REQUIREMENTS = [req.rstrip() for req in REQUIREMENTS]


setup(
    name='karaboFAI',
    version="0.1.0",
    author='Jun Zhu',
    author_email='jun.zhu@xfel.eu',
    description='Azimuthal integration tool',
    long_description='Offline and online data analysis and visualization tool '
                     'for azimuthal integration of different data acquired '
                     'with various detectors at European XFEL.',
    url='',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'fxe-gui=karaboFAI.fxeFAI.fxe_gui:fxe_gui',
            'spb-gui=karaboFAI.spbFAI.spb_gui:spb_gui'
        ],
    },
    package_data={},
    install_requires=REQUIREMENTS,
)
