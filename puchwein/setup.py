from setuptools import setup, find_packages

setup(
    name='puchwein',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'sklearn'
    ],
    author='Beth Delaney',
    author_email='beth_delaney@outlook.com',
    description='Python implementation of the Puchwein algorithm for selecting calibration samples',
    url='https://github.com/your-github-repo',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)