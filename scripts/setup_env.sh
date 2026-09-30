#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Python and all library versions are shared without local uv project scaffolding.
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.11.11 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:3] == (3, 11, 11), "Use Python 3.11.11 for the shared environment"'
uv pip sync --python .venv/bin/python requirements.txt
if [ ! -f .env ]; then
  cp config/env.sample .env
fi
printf 'Environment ready. Set API keys in .env, then run .venv/bin/python app.py --demo\n'
