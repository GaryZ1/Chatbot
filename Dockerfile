# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels ".[anthropic,api,qdrant,parsing,redis]"

FROM base AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=build /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels
COPY configs ./configs
USER app
EXPOSE 8000
ENV CHATBOT_CONFIG=/app/configs/pipeline.yaml
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "chatbot.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
