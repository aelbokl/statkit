from setuptools import setup, find_packages

setup(
    name='analytikit',
    version='0.1.0',
    author='Ahmed Elbokl and Maha Sabry',
    author_email='ahmed.elbokl@med.asu.edu.eg',
    description='A collection of helper functions for data cleaning, statistics and analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='', # Optional
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'pingouin',
        'statsmodels',
        'scikit-posthocs',
        'seaborn',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)