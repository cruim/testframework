from setuptools import setup

setup(
    name="testframework",
    description="Library for testing ML models",
    version="0.1.0",
    py_modules=["testframework"],
    install_requires=[
        "Click"
    ],
    entry_points={
        "console_scripts": [
            "testframework = testframework:cli",
        ],
    },
    license="MIT",
    author="Daniil Arashkevich",
    maintainer_email="darashkevich@inno.tech",
    author_email="daniil.arashkevich@gmail.com",
)
