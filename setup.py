from setuptools import setup, find_packages

setup(
    name='dlpkg',
    version='0.1.0',
    author='Dazhong Li',
    author_email='dazhong.li@arup.com',
    description='basic calculation',
    packages=['src'],
    install_requires=[
        'pandas>1.0',
        'Plotly',
        'matplotlib>3.0',
        'scipy>=1.4',
        'tqdm',
    ]
)
