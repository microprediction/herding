import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="herding",
    version="0.0.1",
    description="Kernel herding",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/herding",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    packages=["herding",
              "herding.animation",
              "herding.gaussianherding",
              "herding.gaussiankernel"
              ],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=['numpy','randomcov'],
    entry_points={
        "console_scripts": [
            "herding=herding.__main__:main",
        ]
    },
)