import sys

required_verion = (3,6)
if sys.version_info < required_verion:
    raise ValueError('needs at least python {}! You are trying to install it under python {}'.format('.'.join(str(i) for i in required_verion), sys.version))

# import ez_setup
# ez_setup.use_setuptools()

from setuptools import setup
# from distutils.core import setup
setup(
    name="nesdis_gml_synergy",
    version="0.1",
    packages=['nesdis_gml_synergy'],
    author="Hagen Telg",
    author_email="hagen@hagnet.net",
    description="Everything around the the nesdis_gml_synergy project",
    license="MIT",
    keywords="nesdis_gml_synergy",
    url="https://github.com/hagne/nesdis_gml_synergy",
    scripts=['scripts/scrape_sat', 
             'scripts/goes_aws_scaper_surfrad',
             'scripts/goesscraper', 
             # 'scripts/hrrr_smoke2gml'
             ],
    # install_requires=['numpy','pandas'],
    # extras_require={'plotting': ['matplotlib'],
    #                 'testing': ['scipy']},
    # test_suite='nose.collector',
    # tests_require=['nose'],
)