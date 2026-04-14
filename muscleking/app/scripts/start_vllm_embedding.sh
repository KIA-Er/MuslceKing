#!/bin/bash
set -e

echo "🚀 Starting vLLM Embedding Server (BGE-M3)..."

# 检查 GPU
echo "📊 Checking GPU availability..."
nvidia-smi || echo "⚠️  Warning: No GPU detected"

# 检查端口
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8001 is already in use"
    echo "🔍 Stopping existing container..."
    docker stop vllm-embedding 2>/dev/null || true
    docker rm vllm-embedding 2>/dev/null || true
fi

# 启动 vLLM Embedding Server
echo "🐳 Starting vLLM container..."
docker run -d \
  --name vllm-embedding \
  --gpus all \
  -p 50001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --restart unless-stopped \
  vllm/vllm:v0.6.4 \
  --model BAAI/bge-m3 \
  --port 8000 \
  --embedding-mode True \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --dtype auto \
  --trust-remote-code

echo "⏳ Waiting for vLLM to start (this may take 1-2 minutes)..."
sleep 30

# 健康检查
for i in {1..10}; do
    if curl -s http://localhost:50001/health >/dev/null 2>&1; then
        echo "✅ vLLM Embedding Server is healthy!"
        echo "📍 Endpoint: http://localhost:50001/v1"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Health check failed after 10 attempts"
        echo "📋 Container logs:"
        docker logs --tail 50 vllm-embedding
        exit 1
    fi
    echo "⏳ Attempt $i/10..."
    sleep 10
done

echo "✅ vLLM Embedding Server started successfully!"
