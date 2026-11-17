# Imagen de inferencia (CPU). Requiere haber generado models/final/ con src/07_exportar.py.
FROM python:3.11-slim

WORKDIR /srv
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy fastapi "uvicorn[standard]" pydantic

COPY src/models.py src/models.py
COPY app/ app/
COPY models/final/ models/final/

EXPOSE 8000
# 2 workers ≈ 2 vCPU (c7g.large); cada worker usa 1 hilo de torch
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
