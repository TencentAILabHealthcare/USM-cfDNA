#!/usr/bin/env bash
set -euo pipefail

which python

pip install --upgrade --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'avail=', torch.cuda.is_available())"

pip install --no-cache-dir \
  -f https://data.pyg.org/whl/torch-2.5.1+cu124.html \
  pyg-lib==0.4.0+pt25cu124 \
  torch-scatter==2.1.2+pt25cu124 \
  torch-sparse==0.6.18+pt25cu124 \
  torch-cluster==1.6.3+pt25cu124 \
  torch-spline-conv==1.2.2+pt25cu124 \
  torch-geometric==2.6.1

pip install --no-cache-dir --no-build-isolation causal-conv1d==1.5.0.post8
pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1
pip install --no-cache-dir --no-build-isolation ml-collections==0.1.1
pip install --no-cache-dir --no-build-isolation deepspeed==0.14.4
pip install --no-cache-dir --no-build-isolation mamba-ssm==2.2.4
pip install --no-cache-dir pytorch-lightning==2.5.1.post0 torchmetrics==1.7.3
rm -rf apex
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 24.04.01
pip install -r requirements.txt
python setup.py install --cpp_ext --cuda_ext
cd ..
rm -rf apex

echo "All done."