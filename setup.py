from setuptools import Extension, setup

module = Extension("mysymnmf", sources=['symnmfmodule.c', 'symnmf.c'])
setup(name='pkg',
     version='1.0',
     description='Python wrapper for C extension',
     ext_modules=[module])
