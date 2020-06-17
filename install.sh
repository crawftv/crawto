#!/usr/bin/env bash
set -euo pipefail

pip-compile requirements.in --output-file=requirements.txt
pip install -r requirements.txt
pip-compile test-requirements.in --output-file=test-requirements.txt
pip install -r test-requirements.txt
pip install torch===1.4.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/crawftv/nca.git
