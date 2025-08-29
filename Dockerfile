FROM python:3.10-slim-buster

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies and clean up immediately
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        langchain-huggingface \
        langchain-community \
        langchain-openai \
        langchain-pinecone \
        langchain-core \
    && apt-get remove -y gcc g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

# Now copy the rest of the application
COPY . .

CMD ["python3", "app.py"]