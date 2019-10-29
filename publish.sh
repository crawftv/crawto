#!/bin/bash
rm -r dist
rm -r build
python setup.py sdist bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
