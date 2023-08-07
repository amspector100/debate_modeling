import numpy as np
import os.path
import codecs
from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

# ### Allows installation if cython is not yet installed
# try:
#     from Cython.Build import cythonize
# except ImportError:
#     # create closure for deferred import
#     def cythonize (*args, ** kwargs ):
#         from Cython.Build import cythonize
#         return cythonize(*args, ** kwargs)

ext_modules = [
	Extension(
		"dmsrc.models._truncnorm",
		sources=["dmsrc/models/_truncnorm.pyx"],
	),
]

setup(
	name="dmsrc",
	version=get_version('dmsrc/__init__.py'),
	packages=find_packages(),
	description='Debate modeling',
	long_description=long_description,
	long_description_content_type="text/markdown",
	author='Asher Spector',
	author_email='amspector100@gmail.com',
	url='https://github.com/amspector100/dmsrc',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	ext_modules=cythonize(
		ext_modules,
		compiler_directives={
			"language_level": 3, 
			"embedsignature": True
		},
		annotate=False,
	),
	include_dirs=[np.get_include()],
	python_requires='>=3.6',
    install_requires=[
        "numpy>=1.19",
        "scipy>=1.6.1",
        "pandas>=1.4.2", 
        "cython>=0.29.21",
    ],
    setup_requires=[
        'numpy>=1.19',
    	'setuptools>=58.0',
    	'cython>=0.29.21',
    ]
)