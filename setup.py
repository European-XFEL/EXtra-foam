"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import contextlib
import glob
import multiprocessing as mp
import os
import os.path as osp
import re
import shutil
import sys
import sysconfig
import subprocess
from setuptools import setup, Command, find_packages, Distribution, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as _TestCommand
from distutils.command.clean import clean
from distutils.version import LooseVersion
from distutils.util import strtobool


with open(osp.join(osp.abspath(osp.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()


def find_version():
    with open(osp.join('extra_foam', '__init__.py')) as fp:
        for line in fp:
            m = re.search(r'^__version__ = "(\d+\.\d+\.\d[a-z]*\d*)"', line, re.M)
            if m is None:
                # could be a hotfix
                m = re.search(r'^__version__ = "(\d.){3}\d"', line, re.M)
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
    CMakeExtension("extra_foam"),
]


class BuildExt(build_ext):

    _thirdparty_exec_files = [
        "extra_foam/thirdparty/bin/redis-server",
        "extra_foam/thirdparty/bin/redis-cli"
    ]

    _foam_algo_lib_path = 'extra_foam/algorithms'

    description = "Build the C++ extensions for EXtra-foam"
    user_options = [
        ('use-tbb', None, 'build with intel TBB'),
        ('xtensor-use-tbb', None, 'build xtensor with intel TBB'),
        # https://quantstack.net/xsimd.html
        ('use-xsimd', None, 'build with XSIMD'),
        ('xtensor-use-xsimd', None, 'build xtensor with XSIMD'),
        ('with-tests', None, 'build cpp unittests'),
    ] + build_ext.user_options

    def initialize_options(self):
        super().initialize_options()

        build_serial = strtobool(os.environ.get('BUILD_SERIAL_FOAM', '0'))
        build_para = '0' if build_serial else '1'
        self.use_tbb = strtobool(
            os.environ.get('FOAM_USE_TBB', build_para))
        self.xtensor_use_tbb = strtobool(
            os.environ.get('XTENSOR_USE_TBB', build_para))
        self.use_xsimd = strtobool(
            os.environ.get('FOAM_USE_XSIMD', build_para))
        self.xtensor_use_xsimd = strtobool(
            os.environ.get('XTENSOR_USE_XSIMD', build_para))

        self.with_tests = strtobool(os.environ.get('BUILD_FOAM_TESTS', '0'))

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the "
                               "following extensions: " + ", ".join(
                e.name for e in self.extensions))

        cmake_version = LooseVersion(
            re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        cmake_minimum_version_required = '3.13.0'
        if cmake_version < cmake_minimum_version_required:
            raise RuntimeError(f"CMake >= {cmake_minimum_version_required} "
                               f"is required!")

        # build third-party libraries, for example, Redis
        command = ["./build.sh", "-p", sys.executable]
        subprocess.check_call(command)
        self._move_thirdparty_exec_files()

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        ext_dir = osp.abspath(osp.dirname(self.get_ext_fullpath(ext.name)))
        build_type = 'debug' if self.debug else 'release'
        build_temp = osp.join(os.getcwd(), self.build_temp)
        build_lib = osp.join(os.getcwd(), self.build_lib)
        saved_cwd = os.getcwd()

        cmake_options = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={osp.join(ext_dir, 'extra_foam/algorithms')}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_PREFIX_PATH={os.getenv('CMAKE_PREFIX_PATH')}",
            f"-DBUILD_FOAM_PYTHON=ON",
        ]

        def _opt_switch(x):
            return 'ON' if x else 'OFF'

        cmake_options.append(
            f'-DFOAM_USE_TBB={_opt_switch(self.use_tbb)}')
        cmake_options.append(
            f'-DXTENSOR_USE_TBB={_opt_switch(self.xtensor_use_tbb)}')

        cmake_options.append(
            f'-DFOAM_USE_XSIMD={_opt_switch(self.use_xsimd)}')
        cmake_options.append(
            f'-DXTENSOR_USE_XSIMD={_opt_switch(self.xtensor_use_xsimd)}')

        cmake_options.append(
            f'-DBUILD_FOAM_TESTS={_opt_switch(self.with_tests)}')

        max_jobs = os.environ.get('BUILD_FOAM_MAX_JOBS', str(mp.cpu_count()))
        build_options = ['--', '-j', max_jobs]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        with changed_cwd(self.build_temp):
            # generate build files
            print("-- Running cmake for extra-foam")
            self.spawn(['cmake', ext.source_dir] + cmake_options)
            print("-- Finished cmake for extra-foam")

            # build
            print("-- Running cmake --build for extra-foam")
            self.spawn(['cmake', '--build', '.'] + build_options)
            print("-- Finished cmake --build for extra-foam")

            if self.inplace:
                build_lib = saved_cwd

            try:
                os.makedirs(osp.join(build_lib, self._foam_algo_lib_path))
            except OSError:
                pass

            # placeholder
            # if self.use_tbb or self.xtensor_use_tbb:
            #     self._move_shared_libs('tbb', build_temp, build_lib)

    def _move_thirdparty_exec_files(self):
        for filename in self._thirdparty_exec_files:
            src = filename
            dst = os.path.join(self.build_lib, filename)

            parent_directory = os.path.dirname(dst)
            if not os.path.exists(parent_directory):
                os.makedirs(parent_directory)

            if not os.path.exists(dst):
                self.announce(f"copy {src} to {dst}", level=1)
                shutil.copy(src, dst)

    def _move_shared_libs(self, lib_name, build_temp, build_lib):
        self._move_shared_libs_unix(lib_name, build_temp, build_lib)

    def _move_shared_libs_unix(self, lib_name, build_temp, build_lib):
        if sys.platform == 'darwin':
            lib_pattern = f"lib{lib_name}*.dylib"
        else:
            lib_pattern = f"lib{lib_name}*.so*"

        libs = glob.glob(lib_pattern)

        if not libs:
            raise Exception(f"Could not find shared library with pattern: "
                            f"{lib_pattern}")
        # TODO: deal with libraries with symlinks
        for lib in libs:
            shutil.move(osp.join(build_temp, lib),
                        osp.join(build_lib, self._foam_algo_lib_path, lib))


class TestCommand(_TestCommand):
    def _get_build_dir(self, dirname):
        version = sys.version_info
        return f"{dirname}.{sysconfig.get_platform()}-{version[0]}.{version[1]}"

    def run(self):
        # build and run cpp test
        build_temp = osp.join('build', self._get_build_dir('temp'))
        with changed_cwd(build_temp):
            self.spawn(['make', 'ftest'])

        # run Python test
        import pytest
        errno = pytest.main(['extra_foam'])
        sys.exit(errno)


class BenchmarkCommand(Command):

    description = "run benchmark after in-place build"

    user_options = []

    # TODO: improve

    def initialize_options(self):
        """Override."""
        pass

    def finalize_options(self):
        """Override."""
        pass

    def run(self):
        self.spawn(['python', 'benchmarks/benchmark_imageproc.py'])
        self.spawn(['python', 'benchmarks/benchmark_geometry.py'])
        self.spawn(['python', 'benchmarks/benchmark_statistics.py'])


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name='EXtra-foam',
    version=find_version(),
    author='Jun Zhu',
    author_email='da-support@xfel.eu',
    description='Online analysis and monitoring tool at European XFEL',
    long_description=long_description,
    url='',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'extra-foam=extra_foam.services:application',
            'extra-foam-special-suite=extra_foam.special_suite.services:application',
            'extra-foam-kill=extra_foam.services:kill_application',
            'extra-foam-stream=extra_foam.services:stream_file',
            'extra-foam-redis-cli=extra_foam.services:start_redis_client',
            'extra-foam-monitor=extra_foam.web.monitor:web_monitor'
        ],
    },
    ext_modules=ext_modules,
    tests_require=['pytest'],
    cmdclass={
        'clean': clean,
        'build_ext': BuildExt,
        'test': TestCommand,
        'benchmark': BenchmarkCommand,
    },
    distclass=BinaryDistribution,
    package_data={
        'extra_foam': [
            'gui/icons/*.png',
            'gui/icons/*.jpg',
            'geometries/*.h5',
            'geometries/*.geom',
            'configs/*.yaml',
        ]
    },
    install_requires=[
        'numpy',
        'pybind11',
        'scipy',
        'msgpack',
        'msgpack-numpy',
        'pyzmq',
        'pyFAI',
        'PyQt5',
        'PyQt5-sip',
        'EXtra-data',
        'EXtra-geom',
        'karabo-bridge',
        'hiredis',
        'redis',
        'psutil',
        'imageio',
        'Pillow',
        'pyyaml',
        "metropc @ git+ssh://git@git.xfel.eu:10022/karaboDevices/metropc.git@high_high_water_mark",
        "qscintilla",
        "libcst",
        # These dependencies are not directly used, but are needed to satisfy
        # pip's resolver:
        'pygments',
        'jinja2',
        'decorator',
        'pexpect',
        'pygments',
        'toolz',
        'packaging',
        'cffi'
    ],
    extras_require={
        'docs': [
            'sphinx',
            'nbsphinx',
            'ipython',  # For nbsphinx syntax highlighting
            'sphinx_rtd_theme',
            'sphinxcontrib_github_alt'
        ],
        'test': [
            'pytest',
            'pytest-cov',
        ],
        'web': [
            'dash>=1.1.0',
        ],
    },
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
