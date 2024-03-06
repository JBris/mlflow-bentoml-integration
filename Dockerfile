ARG PYTHON_TAG

FROM python:${PYTHON_TAG} as base

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/downloaded_packages

FROM base as builder

ARG PYTHON_POETRY_VERSION

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock ./

RUN pip install poetry==${PYTHON_POETRY_VERSION} \
    && poetry install --no-root \
    && rm -rf $POETRY_CACHE_DIR 

FROM base as runtime

ENV VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:$PATH"

COPY ./ ./

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
