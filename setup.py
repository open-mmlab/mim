from setuptools import find_packages, setup  # type: ignore


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'mim/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name='openmim',
    version=get_version(),
    description='MIM Installs OpenMMLab packages',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/open-mmlab/mim',
    author='MIM Authors',
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
