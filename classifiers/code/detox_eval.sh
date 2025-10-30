#!/bin/bash

echo "🔄 正在切换到 Conda 环境 zhtox_swift..."
eval "$(conda shell.bash hook)"
conda activate zhtox_swift

# 检查 conda 环境是否激活成功
if [[ "$CONDA_DEFAULT_ENV" == "zhtox_swift" ]]; then
  echo "✅ Conda 环境 zhtox_swift 已成功激活！"
else
  echo "❌ Conda 环境激活失败！当前环境为：$CONDA_DEFAULT_ENV"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python detox_eval.py --folder /home/ToxiRewriteCN/finetuning_llama3-8b/eval
