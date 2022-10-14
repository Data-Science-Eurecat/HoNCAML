from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README.md file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "config/requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line]


# Proper setup
setup(
    name="honcaml",
    version="0.1",
    description="Holistic and No Code Auto Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # TODO: Pending code and Pypi URL
    # url="https://ice.eurecat.org/gitlab/big-data/honcaml",
    # download_url="https://pypi.org/project/honcaml/",
    # TODO: Author should be the same as the project leader?
    # author="Cirus Iniesta",
    # author_email="cirus.iniesta@eurecat.org",
    license="BSD License",
    keywords="machine_learning automl nocode",
    # TODO: Exclude docs and tests when official, e.g.
    # find_packages(exclude=["contrib", "docs", "tests"])
    packages=find_packages(),
    data_files=[('config', ['config/test.ini'])],
    # TODO: Set python and dependencies requirements
    # python_requires=">=3.8",
    # install_requires=requirements,
    entry_points={"console_scripts": ["honcaml=honcaml.__main__:cli"]},
)
