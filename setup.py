import os
import pathlib

import pkg_resources
from setuptools import setup, find_packages

PKG_NAME = "rmrl"
VERSION = "0.0.1"
EXTRAS = {}


def _read_file(fname):
    # this_dir = os.path.abspath(os.path.dirname(__file__))
    # with open(os.path.join(this_dir, fname)) as f:
    with pathlib.Path(fname).open() as fp:
        return fp.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        main = [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]
    plugins = []
    for plugin in os.listdir("allenact_plugins"):
        req_path = os.path.join("allenact_plugins", plugin, "extra_requirements.txt")
        if os.path.exists(req_path):
            with pathlib.Path(req_path).open() as fp:
                plugins += [
                    str(requirement)
                    for requirement in pkg_resources.parse_requirements(fp)
                ]
    return main + plugins


def _fill_extras(extras):
    if extras:
        extras["all"] = list(set([item for group in extras.values() for item in group]))
    return extras


setup(
    name=PKG_NAME,
    version=VERSION,
    author=f"{PKG_NAME} Developers",
    # url='http://github.com/',
    description="research project",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=["Deep Learning", "Reinforcement Learning"],
    license="MIT",
    packages=find_packages(include=f"{PKG_NAME}.*"),
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            # 'cmd_tool=mylib.subpkg.module:main',
        ]
    },
    install_requires=_read_install_requires(),
    extras_require=_fill_extras(EXTRAS),
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
    ],
)
