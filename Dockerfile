# FROM python:3.9-slim
FROM face:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    unixodbc \
    unixodbc-dev \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Add Microsoft repo
RUN curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/10/prod buster main" \
      > /etc/apt/sources.list.d/mssql-release.list

# Install MS ODBC driver
RUN apt-get update \
 && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python setup
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./src /app/src
COPY .env /app/.env

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

CMD ["uvicorn", "src.api.endpoint:app", "--host", "0.0.0.0", "--port", "5001", "--log-level", "debug"]
