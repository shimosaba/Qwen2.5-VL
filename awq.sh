#!/bin/bash

lmdeploy lite auto_awq \
   "Qwen/Qwen2.5-VL-3B-Instruct" \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --w-bits 4 \
  --w-group-size 128 \
  --batch-size 1 \
  --work-dir "/workspace/model_3b_4bit"
