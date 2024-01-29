FROM python:3.11-slim

WORKDIR /usr/src/app

COPY ./tp.py .
COPY ./requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python3", "tp.py" ]
