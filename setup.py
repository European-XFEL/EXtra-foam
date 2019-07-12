import os
import os.path as osp
import platform
import re
import shutil
import sys
import subprocess
from setuptools import setup, find_packages, Distribution, Extension
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean
from distutils.version import LooseVersion
from distutils.util import strtobool


def find_version():
    with open(os.path.join('karaboFAI', '__init__.py')) as fp:
        for line in fp:
            m = re.search(r'^__version__ = "(\d+\.\d+\.\d[a-z]*\d*)"', line, re.M)
            if m is not None:
                return m.group(1)
        raise RuntimeError("Unable to find version string.")


class CMakeExtension(Extension):
    def __init__(self, name, source_dir=''):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


ext_modules = [
    CMakeExtension("karaboFAI"),
]


class BuildExt(build_ext):

    _thirdparty_files = [
        "karaboFAI/thirdparty/bin/redis-server",
    ]

    description = "Build the C++ extensions for karaboFAI"
    user_options = build_ext.user_options.extend([
        ('with-tbb', None, 'build xtensor with intel TBB'),
        # https://quantstack.net/xsimd.html
        ('with-xsimd', None, 'build xtensor with XSIMD'),
    ])

    def initialize_options(self):
        super().initialize_options()

        self.with_tbb = strtobool(os.environ.get('FAI_WITH_TBB', '0'))
        self.with_xsimd = strtobool(os.environ.get('FAI_WITH_XSIMD', '0'))

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the "
                               "following extensions: " + ", ".join(
                e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_cmake(ext)

        # build third-party libraries, for example, Redis
        command = ["./build.sh", "-p", sys.executable]
        subprocess.check_call(command)
        for filename in self._thirdparty_files:
            self._move_file(filename)

    def build_cmake(self, ext):
        ext_dir = osp.abspath(osp.dirname(self.get_ext_fullpath(ext.name)))
        build_type = 'debug' if self.debug else 'release'

        cmake_options = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={osp.join(ext_dir, 'karaboFAI/cpp')}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]

        if self.with_tbb:
            cmake_options.append('-DXTENSOR_USE_TBB=ON')

        if self.with_xsimd:
            cmake_options.append('-DXTENSOR_USE_XSIMD=ON')

        build_options = ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.source_dir] + cmake_options,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_options,
                              cwd=self.build_temp)

        print()  # add an empty line to improve output readability

    def _move_file(self, filename):
        """Move file to the system folder."""
        src = filename
        dst = os.path.join(self.build_lib, filename)

        parent_directory = os.path.dirname(dst)
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)

        if not os.path.exists(dst):
            self.announce(f"copy {src} to {dst}", level=1)
            shutil.copy(src, dst)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


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
            'karaboFAI=karaboFAI.services:application',
            'karaboFAI-kill=karaboFAI.services:kill_application',
            'karaboFAI-stream=karaboFAI.services:stream_file'
        ],
    },
    ext_modules=ext_modules,
    cmdclass={
        'clean': clean,
        'build_ext': BuildExt,
    },
    distclass=BinaryDistribution,
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
        'karabo-data>=0.5.0',
        'karabo-bridge>=0.3.0',
        'toolz',
        'silx>=0.9.0',
        'redis',
        'pyarrow>=0.13.0',
        'psutil',
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
