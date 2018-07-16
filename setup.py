from distutils.core import setup, Extension
import numpy.distutils.misc_util


setup (name = 'cArray',
       version = '1.0',
       description = 'This is a package for improving matrix calculations',
       ext_modules=[
                    Extension('cArray', sources = ['cArray.c'],
                              include_dirs=[numpy.get_include()]),
                    ],
       )