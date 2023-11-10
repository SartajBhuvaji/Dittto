from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'Generate synthetic Tabular Data'

# Read the contents of your README.md file
with open('README.md', 'r', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
