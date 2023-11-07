from setuptools import setup, find_packages

VERSION = '0.1.10'
DESCRIPTION = 'Generate synthetic Tabular Data'
LONG_DESCRIPTION = 'A package with multiple alogrithms to generate synthetic tabular data.'

# Setting up
setup(
    name="dittto",
    version=VERSION,
    author="Sartaj ",
    author_email="sbhuvaji@seattleu.edu",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'tensorflow'],
    keywords=['python', 'synthetic data', 'synthetic data generation', 'tabular data', ' csv',],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)