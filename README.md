# Frequency-Based Image Watermarking Using PyTorch

A **digital image watermarking project** using **frequency domain embedding (DWT)** and deep learning. The project embeds a watermark into a host image using a learned **Embedder** network and extracts it back with frequency-based operations.  

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Architecture](#architecture)  
- [Requirements](#requirements)  
- [Setup](#setup)  
- [Usage](#usage)  
  - [Embedding](#embedding)  
  - [Extraction](#extraction)  
- [Training](#training)  
- [Metrics](#metrics)  
- [Project Structure](#project-structure)  
- [License](#license)  

---

## Overview

This project implements a **robust watermarking system** using **Discrete Wavelet Transform (DWT)** in the frequency domain:

- The **Embedder** network learns to compute the embedding strength (`alpha`) based on the host image content.  
- Watermark is embedded in **LH and HL high-frequency subbands** of the host image.  
- Extraction is done by comparing differences between host and watermarked image in frequency subbands.  
- Goal: Invisible watermarking with high **image quality** and **watermark recoverability**.

---

## Features

- Dynamic **embedding strength** using neural network.  
- Embeds watermark in **frequency domain (DWT)** for better robustness.  
- Supports **grayscale images**.  
- Calculates **PSNR**, **SSIM**, and **Normalized Correlation (NC)** during training.  
- Separate **Embedder** and **Extractor** models for modular usage.  

---

## Architecture

**Embedder:**

- Input: Grayscale host image `[1,1,H,W]`  
- Layers: `Conv2d → ReLU → Conv2d → ReLU → AdaptiveAvgPool → Flatten → Linear → Sigmoid`  
- Output: Alpha (embedding strength in range 0.002 – 0.02)

**Frequency Embedding:**

- Apply **DWT** on host image → obtain `Yl` (low-frequency) and `Yh` (high-frequency: LH, HL, HH)  
- Resize watermark to match LH/HL subbands  
- Embed watermark: `LH_emb = LH + alpha * wm`  
- Reconstruct watermarked image using **IDWT**

**Extraction:**

- Apply DWT to **host** and **watermarked image**  
- Compute watermark from differences in LH and HL subbands:  
  ```python
  wm_ext = (LH_wm - LH_host + HL_wm - HL_host) / (2*alpha)
