from setuptools import setup, find_packages
from os import path
from multiprocessing import freeze_support

here = path.abspath(path.dirname(__file__))

if __name__ == '__main__':
    freeze_support()
    setup(

        name='sca',

        description='This is a project where side-channel attacks are researched and developed.',

        url='https://gitlab.ewi.tudelft.nl/TI2806/2018-2019/CS/cp19-cs-11/cs-11',

        packages=find_packages(exclude=['.gitlab', 'data', 'doc', 'out', 'tests']),

        python_requires='>= 3.7',

        install_requires=[
            'numpy',
            'click',
            'scipy',
            'pyitlib',
            'tabulate',
            'progressbar2',
            'torch',
            'matplotlib',
            'fastdtw'
        ],

        dependency_links=[
            "https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl",
            "https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl"
        ],

        test_suite='tests',

        extras_require={
            'test': ['unittest'],
        },

    )
