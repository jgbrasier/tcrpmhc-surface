from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='tcrpmhc surface',
   version='0.1.0',
   description='TCR-pMHC binding surface prediction',
   long_description=long_description,
   license="MIT",
   author='Jean-Guillaume Brasier',
   author_email='jbrasier@g.harvard.edu',
   packages=find_packages(include=['tcrpmhc_surface']),#same as name
#    package_dir={'src': 'src/'}
)