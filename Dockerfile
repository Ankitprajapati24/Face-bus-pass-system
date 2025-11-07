FROM python:3.10

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    flask==3.1.2 \
    flask-cors==6.0.1 \
    gunicorn==23.0.0 \
    numpy \
    pillow

RUN pip install --no-cache-dir opencv-python-headless

RUN pip install --no-cache-dir \
    dlib \
    face-recognition \
    face_recognition_models

RUN pip install --no-cache-dir \
    tensorflow \
    deepface \
    keras

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

COPY . .

RUN mkdir -p data/registered_faces data/group_scans data/unpaid_captures

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 180 backend.app:app