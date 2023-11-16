FROM python:3.11.1-slim

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install \
    build-essential \
    netcat \
    libgl1 \
    libgl1-mesa-glx \ 
    libglib2.0-0 \
    -y && apt-get clean
# Install the specified packages
RUN pip install --upgrade -r requirements.txt --no-cache-dir && rm -rf /root/.cache
COPY ./app /app
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]