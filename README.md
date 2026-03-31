
# Gardner Multiple Intelligences Neural System (PyTorch + PyG)

## Overview
This project implements Howard Gardner's Multiple Intelligences as a modular neural system:

Intelligences:
- Logical-Mathematical (MLP)
- Linguistic (Transformer Encoder)
- Spatial (CNN)
- Musical (Audio CNN)
- Bodily-Kinesthetic (Policy Network)
- Interpersonal (Graph Neural Network - PyG)
- Intrapersonal (GRU self-model)
- Naturalistic (Environmental embedding)

## Key Features
- PyTorch + PyTorch Geometric (GNN for interpersonal intelligence)
- Multi-agent simulation (agents = nodes)
- Emergence via graph interactions
- Evolutionary fitness function

## Fitness
Fitness = mean(output) + std(output) - var(output)

## Files
- model.py
- train.py
- colab.ipynb
- README.md

## Install
pip install torch torchvision torchaudio torch-geometric

## Run
python train.py

## Colab
Upload zip → open notebook → run all
