FROM python:3.10-slim

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gcc \
    g++ \
    make \
    cmake \
    git \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install TA-Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && make && make install

# After installing TA-Lib, reduce the size of the installed libraries
RUN strip /usr/local/lib/libta_lib.*

# Set TA-Lib environment variables for pip build
ENV TA_LIBRARY_PATH=/usr/local/lib
ENV TA_INCLUDE_PATH=/usr/local/include
ENV CFLAGS="-I/usr/local/include"
ENV LDFLAGS="-L/usr/local/lib"

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip uninstall -y numpy ta-lib || true
RUN pip install numpy==1.26.4 Cython==0.29.36
RUN pip install ta-lib==0.4.24
RUN pip install --force-reinstall numpy==1.26.4
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Set the port (Render will pass PORT as an environment variable)
ENV PORT=10000

# Expose the port
EXPOSE $PORT

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--timeout-keep-alive", "600"]
