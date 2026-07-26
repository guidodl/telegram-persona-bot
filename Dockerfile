FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY bot ./bot
COPY migrations ./migrations

RUN pip install --no-cache-dir .

ENTRYPOINT ["sh", "-c", "python -m bot.migrate && python -m bot.ingress"]
