from setuptools import setup, find_packages

setup(
    name='annot',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'annot': [
            'data/gcms_lib.csv',
            'data/hmdb.csv',
            'data/lc_lib.csv',
            'data/t3db.csv',
        ],
    },
    include_package_data=True,
    description='Lightweight helpers to annotate untargeted metabolomics data.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Boris Minasenko',
    author_email='boris.minasenko@emory.edu',
    url='https://github.com/BM-Boris/annot',
    install_requires=[
        'numpy',  
        'pandas',  
        'tqdm'
        
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
