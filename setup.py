import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="crawto-crawftv",
    version="0.0.2",
    author="Crawford Collins",
    url="https://github.com/crawftv/crawto",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    test_suite="tests",
    entry_points={"console_scripts": ["crawto-ml = crawto.__main__:main",],},
)
