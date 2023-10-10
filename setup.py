
#from distutils.core import setup 

from setuptools import setup, find_packages

setup(  name="DESReg", 
        version= '0.3.3',
		maintainer='María D. Pérez-Godoy',
        maintainer_email='lperez@ujaen.es',
        author='María D. Pérez-Godoy, Marta Molina Pérez, Francisco Martínez, D. Elizondo, Francisco Charte, Antonio J. Rivera',
        author_email='lperez@ujaen.es, mmp00080@red.ujaen.es, fmartin@ujaen.es, elizondo@dmu.ac.uk, fcharte@ujaen.es, arivera@ujaen.es',
        description= 'DES for Regression',
        long_description='Implementation of Dynamic Ensemble Selection methods for Regression',
        license='MIT', 
        
        
        install_requires=[
            'scikit-learn>=1.2.1',
            'numpy>=1.21.5',
            'scipy>=1.5.2',
        ],
        
        
        classifiers= [
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.8',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],

        python_requires=' >=3 ',

      
        packages=find_packages(),
        include_package_data=True,
		package_data={'': ['dataset/*.csv']},		
    )