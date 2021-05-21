from setuptools import find_packages, setup  # type: ignore

setup(
    name='openmim',
    version='0.1.0',
    description='MIM Install OpenMMLab packages',
    author='OpenMMLab',
    author_email='openmmlab@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'Click==7.1.2',
        'colorama',
        'requests',
        'model-index',
        'pandas',
        'tabulate',
    ],
    entry_points='''
        [console_scripts]
        mim=mim.cli:cli
    ''',
)
