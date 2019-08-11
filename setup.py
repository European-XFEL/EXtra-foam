import contextlib
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


@contextlib.contextmanager
def changed_cwd(dirname):
    oldcwd = os.getcwd()
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(oldcwd)


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
        "karaboFAI/thirdparty/bin/redis-cli"
    ]

    description = "Build the C++ extensions for karaboFAI"
    user_options = [
        ('with-tbb', None, 'build with intel TBB'),
        ('xtensor-with-tbb', None, 'build xtensor with intel TBB'),
        # https://quantstack.net/xsimd.html
        ('with-xsimd', None, 'build with XSIMD'),
        ('xtensor-with-xsimd', None, 'build xtensor with XSIMD'),
        ('with-tests', None, 'build cpp unittests'),
    ] + build_ext.user_options

    def initialize_options(self):
        super().initialize_options()

        self.with_tbb = strtobool(os.environ.get('FAI_WITH_TBB', '1'))
        self.xtensor_with_tbb = strtobool(os.environ.get('XTENSOR_WITH_TBB', '1'))
        self.with_xsimd = strtobool(os.environ.get('FAI_WITH_XSIMD', '1'))
        self.xtensor_with_xsimd = strtobool(os.environ.get('XTENSOR_WITH_XSIMD', '1'))
        self.with_tests = strtobool(os.environ.get('FAI_WITH_TESTS', '0'))

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the "
                               "following extensions: " + ", ".join(
                e.name for e in self.extensions))

        cmake_version = LooseVersion(
            re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < '3.8.0':
            raise RuntimeError("CMake >= 3.8.0 is required!")

        # build third-party libraries, for example, Redis
        command = ["./build.sh", "-p", sys.executable]
        subprocess.check_call(command)
        for filename in self._thirdparty_files:
            self._move_file(filename)

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        ext_dir = osp.abspath(osp.dirname(self.get_ext_fullpath(ext.name)))
        build_type = 'debug' if self.debug else 'release'

        cmake_options = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={osp.join(ext_dir, 'karaboFAI/cpp')}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]

        if self.with_tbb:
            cmake_options.append('-DFAI_WITH_TBB=ON')
        else:
            # necessary to switch from ON to OFF
            cmake_options.append('-DFAI_WITH_TBB=OFF')

        if self.xtensor_with_tbb:
            # cmake option in thirdparty/xtensor
            cmake_options.append('-DXTENSOR_USE_TBB=ON')
        else:
            cmake_options.append('-DXTENSOR_USE_TBB=OFF')

        if self.with_xsimd:
            cmake_options.append('-DFAI_WITH_XSIMD=ON')
        else:
            cmake_options.append('-DFAI_WITH_XSIMD=OFF')

        if self.xtensor_with_xsimd:
            # cmake option in thirdparty/xtensor
            cmake_options.append('-DXTENSOR_USE_XSIMD=ON')
        else:
            cmake_options.append('-DXTENSOR_USE_XSIMD=OFF')

        if self.with_tests:
            cmake_options.append('-DBUILD_FAI_TESTS=ON')
        else:
            cmake_options.append('-DBUILD_FAI_TESTS=OFF')

        build_options = ['--', '-j4']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        with changed_cwd(self.build_temp):
            # generate build files
            print("-- Running cmake for karaboFAI")
            self.spawn(['cmake', ext.source_dir] + cmake_options)
            print("-- Finished cmake for karaboFAI")

            # build
            print("-- Running cmake --build for karaboFAI")
            self.spawn(['cmake', '--build', '.'] + build_options)
            print("-- Finished cmake --build for karaboFAI")

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
            'karaboFAI-stream=karaboFAI.services:stream_file',
            'karaboFAI-redis-cli=karaboFAI.services:start_redis_client',
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
            'gui/icons/*.jpg',
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
        'karabo-data>=0.6.2',
        'karabo-bridge>=0.3.0',
        'toolz>=0.9.0',
        'silx>=0.9.0',
        'redis>=3.2.1',
        'pyarrow>=0.13.0',
        'psutil>=5.6.2',
        'imageio>=2.5.0',
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
        'Development Status :: 4 - Beta',
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
