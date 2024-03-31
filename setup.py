from setuptools import setup, find_packages

setup(
    name='vsk-dl-utils',
    version='0.0.3',
    description='My utils to train deep learning models',
    author='Vladislav Sorokin',
    packages=find_packages(include=['vsk_dl_utils', 'vsk_dl_utils.*']),
    install_requires=['torch==2.2.1',
                      'torchvision==0.17.1',
                      'torchmetrics>=0.11.4',
                      'GitPython==3.1.27',
                      'pytweening',
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.10',
    ],
)
