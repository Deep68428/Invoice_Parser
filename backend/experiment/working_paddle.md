# Clean up
rm -rf .venv

# Create updated pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "invoice"
version = "0.1.0"
description = "PaddleOCR-VL Invoice Processing"
readme = "README.md"
requires-python = "==3.11.*"
dependencies = []
EOF

# Create venv with Python 3.11
uv venv --python 3.11
source .venv/bin/activate

# Install packages in specific order to avoid conflicts
uv pip install numpy==1.26.4
uv pip install pillow==10.4.0

# Install PyTorch for CUDA 12.1 (more stable)
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Install PaddlePaddle
uv pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Install transformers and related packages
uv pip install transformers==4.46.0 accelerate==1.2.1 sentencepiece==0.2.0 protobuf==5.29.2

# Install PaddleOCR
uv pip install "paddleocr[doc-parser]"

# Install safetensors
uv pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# Test
python test_paddleocr.py output/00035_page_1.jpg