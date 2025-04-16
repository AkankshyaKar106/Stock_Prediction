FROM python:3.10-slim

# Install system dependencies for TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && sed -i 's/fprintf( out, prefix );/fprintf( out, "%s", prefix );/' src/tools/gen_code/gen_code.c \
    && ./configure --prefix=/usr/local CFLAGS="-Wno-format-security" \
    && make -j1 \
    && sudo make install \
    && sudo ldconfig \
    && cd .. && rm -rf ta-lib

# Set environment variables for TA-Lib installation
ENV TA_LIBRARY_PATH=/usr/local/lib
ENV TA_INCLUDE_PATH=/usr/local/include

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --no-binary :all: ta-lib==0.4.32

# Expose port and run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
