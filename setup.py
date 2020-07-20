from setuptools import setup, find_packages
import os

__version__ = '0.0.1'


with open(os.path.join(path.abspath(path.dirname(__file__)), 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().split('\n')

setup(
    name='models_from_scratch',
    version=__version__,
    description='Machine Learning models from scratch for a better understanding.',
    url='https://github.com/tommywatts/models_from_scratch',
    packages=['models'],
    author='Tommy Watts',
    install_requires=[x.strip() for x in requirements],
    setup_requires=['numpy']
)