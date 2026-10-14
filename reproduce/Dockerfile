# A container is the strongest form of "works on my machine is not an excuse":
# the only inputs are this repository and PyPI.
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[dev]"

# The full check: test suite + known-topology demo + independent recomputation
# of every published number from the committed raw records.
CMD ["sh", "reproduce/run_all.sh"]
