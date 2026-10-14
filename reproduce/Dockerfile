FROM python:3.12-slim
WORKDIR /app
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["python", "reproduce/reproduce_v2.py"]
