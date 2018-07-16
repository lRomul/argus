import os
import io
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = read('README.rst')

VERSION = find_version('argus', '__init__.py')

setup(
    # Metadata
    name='pytorch-argus',
    version=VERSION,
    author='Ruslan Baikulov',
    author_email='ruslan1123@gmail.com',
    url='https://github.com/lRomul/argus',
    description='Easy high-level library for training neural networks in PyTorch.',
    long_description=readme,
    license='MIT',

    # Package info
    packages=find_packages(),

    zip_safe=True,

    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=['torch>=0.4.0'],
)
