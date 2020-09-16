import sys

required_verion = (3,6)
if sys.version_info < required_verion:
    raise ValueError('mie-py needs at least python {}! You are trying to install it under python {}'.format('.'.join(str(i) for i in required_verion), sys.version))

# import ez_setup
# ez_setup.use_setuptools()

from setuptools import setup
# from distutils.core import setup
setup(
    name="sp02",
    version="0.1",
    packages=['sp02'],
    author="Hagen Telg",
    author_email="hagen@hagnet.net",
    description="Everything around the sp02 instrument",
    license="MIT",
    keywords="sp02",
    url="https://github.com/hagne/sp02",
    # install_requires=['numpy','pandas'],
    # extras_require={'plotting': ['matplotlib'],
    #                 'testing': ['scipy']},
    # test_suite='nose.collector',
    # tests_require=['nose'],
)