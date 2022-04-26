FROM python:3.10-slim

RUN mkdir -p /usr/link
WORKDIR /usr/link

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY link.py .
COPY historical_figures_5k.parquet .


CMD ["python", "link.py"]
