# Temporal Synergy Architecture

**Model Name:** Temporal Synergy Classifier  
**Type:** Hybrid Architecture (TCN + Dilated RNN)  
**Performance:** ROC-AUC: 0.669 ± 0.108, PR-AUC: 0.121 ± 0.048 (Best ROC-AUC among all models)

---

## Overview

The Temporal Synergy model is a hybrid architecture that combines the strengths of Temporal Convolutional Networks (TCN) and Dilated Recurrent Neural Networks (Dilated RNN). It processes temporal sequences through two complementary stages:

1. **TCN Stage:** Extracts local temporal patterns using dilated convolutions
2. **Dilated RNN Stage:** Models sequential dependencies from TCN features
3. **Feature Fusion:** Combines multi-scale features from both stages
4. **Attention Pooling:** Provides weighted aggregation for final prediction

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TEMPORAL SYNERGY ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

Input: [B, T, F]
  │
  │  (B = batch size, T = sequence length, F = input features)
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 1: TCN PROCESSING                            │
│                                                                              │
│  Input Projection: [B, T, F] → [B, F, T] → [B, H, T]                        │
│  │                                                                           │
│  ├─► TCN Block 1 (dilation=1)  ──┐                                         │
│  ├─► TCN Block 2 (dilation=2)  ──┤                                         │
│  ├─► TCN Block 3 (dilation=4)  ──┤  Sequential Processing                   │
│  └─► TCN Block 4 (dilation=8)  ──┘                                         │
│                                                                              │
│  Output: [B, H, T] → [B, T, H]                                              │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  │  TCN Features: [B, T, H]
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 2: DILATED RNN PROCESSING                         │
│                                                                              │
│  TCN Output: [B, T, H]                                                       │
│  │                                                                           │
│  ├─► Dilated RNN Layer 1 (dilation=1)  ──┐                                 │
│  │   └─► Output 1: [B, T, H]              │                                 │
│  │                                        │                                 │
│  ├─► Dilated RNN Layer 2 (dilation=2)  ──┤  Hierarchical Processing         │
│  │   └─► Output 2: [B, T, H]              │  (Each layer uses previous      │
│  │                                        │   layer output as input)         │
│  ├─► Dilated RNN Layer 3 (dilation=4)  ──┤                                 │
│  │   └─► Output 3: [B, T, H]              │                                 │
│  │                                        │                                 │
│  └─► Dilated RNN Layer 4 (dilation=8)  ──┘                                 │
│      └─► Output 4: [B, T, H]                                                │
│                                                                              │
│  RNN Outputs: [B, T, H] × 4                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  │  TCN Features: [B, T, H]
  │  RNN Features: [B, T, H] × 4
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FEATURE FUSION                                    │
│                                                                              │
│  Concatenate: [B, T, H] + [B, T, H] × 4 = [B, T, H×5]                       │
│  │                                                                           │
│  ├─► Linear(H×5 → H)                                                        │
│  ├─► ReLU                                                                    │
│  ├─► Dropout                                                                 │
│  └─► Linear(H → H)                                                           │
│                                                                              │
│  Fused Features: [B, T, H]                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  │  Fused Features: [B, T, H]
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ATTENTION POOLING                                   │
│                                                                              │
│  Attention Scores: [B, T, H] → [B, T, 1]                                     │
│  │                                                                           │
│  ├─► Apply Mask (for variable-length sequences)                              │
│  ├─► Softmax                                                                 │
│  └─► Weighted Sum: Σ(attention_weights × features)                           │
│                                                                              │
│  Pooled Features: [B, H]                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  │  Pooled Features: [B, H]
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLASSIFICATION HEAD                                  │
│                                                                              │
│  Linear(H → 1)                                                               │
│                                                                              │
│  Output: [B, 1] (Logits)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Architecture

### Stage 1: TCN Block Structure

Each TCN block uses dilated causal convolutions with residual connections:

```
┌─────────────────────────────────────────────────────────────────┐
│                         TCN BLOCK                                │
│                                                                  │
│  Input: [B, H, T]                                                │
│    │                                                              │
│    ├────────────────────────────────────────────────┐           │
│    │                                                │           │
│    ▼                                                │           │
│  ┌──────────────────────────────────────┐          │           │
│  │  Causal Conv1d (kernel=k, dilation=d) │          │           │
│  │  └─► [B, H, T]                        │          │           │
│  └──────────────────────────────────────┘          │           │
│    │                                                │           │
│    ▼                                                │           │
│  ┌──────────────────────────────────────┐          │           │
│  │  ReLU                                 │          │           │
│  └──────────────────────────────────────┘          │           │
│    │                                                │           │
│    ▼                                                │           │
│  ┌──────────────────────────────────────┐          │           │
│  │  Dropout                             │          │           │
│  └──────────────────────────────────────┘          │           │
│    │                                                │           │
│    ▼                                                │           │
│  ┌──────────────────────────────────────┐          │           │
│  │  Causal Conv1d (kernel=k, dilation=d) │          │           │
│  │  └─► [B, H, T]                        │          │           │
│  └──────────────────────────────────────┘          │           │
│    │                                                │           │
│    ▼                                                │           │
│  ┌──────────────────────────────────────┐          │           │
│  │  ReLU                                 │          │           │
│  └──────────────────────────────────────┘          │           │
│    │                                                │           │
│    ▼                                                │           │
│  ┌──────────────────────────────────────┐          │           │
│  │  Dropout                             │          │           │
│  └──────────────────────────────────────┘          │           │
│    │                                                │           │
│    └────────────────────────────────────┐          │           │
│                                          │          │           │
│  Input: [B, H, T] ───────────────────────┼──────────┘           │
│                                          │                      │
│                                          ▼                      │
│                                    Residual Addition            │
│                                          │                      │
│                                          ▼                      │
│                                    Output: [B, H, T]            │
└─────────────────────────────────────────────────────────────────┘

Key Properties:
- Causal Convolution: Only uses past information (no future leakage)
- Dilation: Exponentially increases (1, 2, 4, 8) to capture multi-scale patterns
- Residual Connection: Helps with gradient flow and training stability
```

### Stage 2: Dilated RNN Layer Structure

Each Dilated RNN layer processes sequences with temporal dilation:

```
┌─────────────────────────────────────────────────────────────────┐
│                      DILATED RNN LAYER                          │
│                                                                  │
│  Input: [B, T, H]                                                │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Temporal Dilation (subsample by d)   │                       │
│  │  └─► [B, T/d, H]                     │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  RNN Cell (LSTM or GRU)               │                       │
│  │  └─► [B, T/d, H]                      │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Upsample to Original Length         │                       │
│  │  └─► [B, T, H]                       │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Layer Normalization                  │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Dropout                             │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  Output: [B, T, H]                                                │
└─────────────────────────────────────────────────────────────────┘

Key Properties:
- Temporal Dilation: Processes every d-th time step to capture long-range dependencies
- Hierarchical Processing: Each layer uses the previous layer's output as input
- Dilation Pattern: Exponentially increases (1, 2, 4, 8) matching TCN dilations
```

### Feature Fusion Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│                         FEATURE FUSION                          │
│                                                                  │
│  Inputs:                                                         │
│    - TCN Output: [B, T, H]                                       │
│    - RNN Output 1: [B, T, H]                                     │
│    - RNN Output 2: [B, T, H]                                     │
│    - RNN Output 3: [B, T, H]                                     │
│    - RNN Output 4: [B, T, H]                                     │
│                                                                  │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Concatenate along feature dimension │                       │
│  │  └─► [B, T, H×5]                     │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Linear(H×5 → H)                     │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  ReLU                                 │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Dropout                             │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Linear(H → H)                       │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  Fused Features: [B, T, H]                                        │
└─────────────────────────────────────────────────────────────────┘

Purpose:
- Combines complementary features from TCN (local patterns) and RNN (sequential dependencies)
- Learns optimal feature combination through learned linear transformations
```

### Attention Pooling Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│                        ATTENTION POOLING                         │
│                                                                  │
│  Input: [B, T, H] (Fused Features)                               │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Linear(H → H)                       │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Tanh                                 │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Linear(H → 1)                        │                       │
│  │  └─► [B, T, 1] (Attention Scores)     │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Apply Mask (mask padding positions)  │                       │
│  │  └─► Set padding to -inf              │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Softmax (normalize attention scores) │                       │
│  │  └─► [B, T, 1] (Attention Weights)    │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────────────────────────┐                       │
│  │  Weighted Sum:                       │                       │
│  │  Σ(attention_weights × features)      │                       │
│  │  └─► [B, H]                          │                       │
│  └──────────────────────────────────────┘                       │
│    │                                                              │
│    ▼                                                              │
│  Pooled Features: [B, H]                                          │
└─────────────────────────────────────────────────────────────────┘

Purpose:
- Selectively aggregates important time steps
- Handles variable-length sequences through masking
- Learns which temporal regions are most informative for prediction
```

---

## Data Flow Through the Network

### Complete Forward Pass

```
Input Sequence: [B, T, F]
  │
  │  Example: B=32, T=3600, F=10 (5 signals + 5 masks)
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Input Projection                                               │
│  Conv1d(F → H): [B, T, F] → [B, F, T] → [B, H, T]              │
│  Example: [32, 3600, 10] → [32, 64, 3600]                       │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  TCN Stage (4 blocks, dilations: 1, 2, 4, 8)                    │
│  [32, 64, 3600] → [32, 64, 3600]                                │
│  Transpose: [32, 64, 3600] → [32, 3600, 64]                     │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Dilated RNN Stage (4 layers, dilations: 1, 2, 4, 8)            │
│  Layer 1: [32, 3600, 64] → [32, 3600, 64]                      │
│  Layer 2: [32, 3600, 64] → [32, 3600, 64]                      │
│  Layer 3: [32, 3600, 64] → [32, 3600, 64]                      │
│  Layer 4: [32, 3600, 64] → [32, 3600, 64]                      │
│                                                                  │
│  Outputs: 4 × [32, 3600, 64]                                    │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Feature Fusion                                                 │
│  Concatenate: [32, 3600, 64] + 4×[32, 3600, 64]                │
│  = [32, 3600, 320] (64 × 5)                                     │
│  Linear(320 → 64): [32, 3600, 320] → [32, 3600, 64]            │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Attention Pooling                                              │
│  Attention Scores: [32, 3600, 64] → [32, 3600, 1]              │
│  Weighted Sum: [32, 3600, 64] × [32, 3600, 1] → [32, 64]       │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Classification                                                 │
│  Linear(64 → 1): [32, 64] → [32, 1]                             │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
Output Logits: [B, 1]
  Example: [32, 1]
```

---

## Architecture Design Rationale

### Why Combine TCN and Dilated RNN?

1. **Complementary Strengths:**
   - **TCN:** Efficiently extracts local temporal patterns using dilated convolutions
   - **Dilated RNN:** Models sequential dependencies and long-range temporal relationships
   - **Combination:** Captures both local patterns and global dependencies

2. **Multi-Scale Feature Extraction:**
   - TCN blocks with dilations (1, 2, 4, 8) capture patterns at different temporal scales
   - Dilated RNN layers with matching dilations process these features hierarchically
   - Feature fusion combines multi-scale representations from both stages

3. **Hierarchical Processing:**
   - TCN stage: Bottom-up feature extraction (local → global)
   - Dilated RNN stage: Sequential refinement (coarse → fine)
   - Each RNN layer refines features from the previous layer

### Key Design Decisions

1. **Matching Dilations:**
   - TCN and RNN stages use the same dilation pattern (1, 2, 4, 8)
   - Ensures alignment between extracted features and sequential processing

2. **Feature Fusion:**
   - Concatenates TCN output with all RNN layer outputs
   - Allows the model to learn optimal combination of features
   - Preserves information from all processing stages

3. **Attention Pooling:**
   - Learns which time steps are most important for prediction
   - Handles variable-length sequences naturally
   - More flexible than mean pooling

---

## Hyperparameters

### Default Configuration

```yaml
input_dim: 10              # 5 signals + 5 masks
hidden_dim: 64             # Hidden dimension for all layers
tcn_levels: 4              # Number of TCN blocks
rnn_layers: 4              # Number of Dilated RNN layers
cell_type: "lstm"         # RNN cell type ("lstm" or "gru")
kernel_size: 3             # Convolution kernel size
dropout: 0.2               # Dropout rate
use_attention: True       # Use attention pooling (vs mean pooling)
```

### Dilation Pattern

- **TCN Blocks:** Dilation = 2^i for i ∈ [0, tcn_levels-1]
  - Example (4 levels): [1, 2, 4, 8]
  
- **Dilated RNN Layers:** Dilation = 2^i for i ∈ [0, rnn_layers-1]
  - Example (4 layers): [1, 2, 4, 8]

### Receptive Field

With 4 TCN levels and 4 RNN layers (both with dilations [1, 2, 4, 8]):

- **TCN Receptive Field:** ~(2^4 - 1) × kernel_size = 15 × 3 = 45 time steps
- **RNN Receptive Field:** Processes every 8th time step at the highest level
- **Combined:** Can capture patterns spanning hundreds of time steps

---

## Performance Characteristics

### Strengths

1. **Best ROC-AUC Performance:** 0.669 ± 0.108 (ranked #1 among all models)
2. **Multi-Scale Pattern Recognition:** Captures both local and global temporal patterns
3. **Robust Feature Extraction:** Combines complementary architectures
4. **Flexible Sequence Handling:** Attention pooling adapts to variable-length sequences

### Limitations

1. **Higher Variability:** Standard deviation of 0.108 indicates sensitivity to fold-specific characteristics
2. **Lower PR-AUC:** 0.121 compared to TCN's 0.140 (TCN has better precision-recall balance)
3. **Computational Complexity:** More parameters and operations than single-architecture models
4. **Memory Usage:** Requires storing intermediate features from both stages

### Comparison with Base Models

| Metric | Temporal Synergy | TCN | Dilated RNN |
|--------|-----------------|-----|-------------|
| ROC-AUC | **0.669** | 0.667 | 0.657 |
| PR-AUC | 0.121 | **0.140** | **0.147** |
| Variability (ROC std) | 0.108 | 0.096 | 0.095 |

**Insight:** Temporal Synergy achieves the best ROC-AUC by combining TCN and RNN, but individual models (TCN, Dilated RNN) achieve better PR-AUC, suggesting the fusion may benefit from further optimization.

---

## Implementation Details

### Key Components

1. **TCNBlock:** Causal dilated convolution with residual connection
2. **DilatedRNNLayer:** RNN with temporal dilation for long-range dependencies
3. **Feature Fusion:** Learned combination of multi-scale features
4. **Attention Pooling:** Mask-aware weighted aggregation

### Training Considerations

- **Batch Size:** 32 (standard, no reduction needed)
- **Memory:** Moderate (less than attention/transformer models)
- **Training Time:** Longer than single-architecture models due to two-stage processing
- **Gradient Flow:** Residual connections in TCN and hierarchical RNN processing help maintain gradients

---

## Visualization for Draw.io

### Recommended Layout

1. **Top Section:** Input → TCN Stage (horizontal flow)
2. **Middle Section:** TCN Output → Dilated RNN Stage (horizontal flow with vertical stacking for RNN layers)
3. **Bottom Section:** Feature Fusion → Attention Pooling → Classification (horizontal flow)

### Color Coding Suggestions

- **Input/Output:** Blue
- **TCN Components:** Green
- **RNN Components:** Orange
- **Fusion/Attention:** Purple
- **Classification:** Red

### Key Connections to Highlight

1. **TCN → RNN:** Shows how TCN features feed into RNN processing
2. **RNN Hierarchical:** Shows how each RNN layer uses previous layer output
3. **Fusion Concatenation:** Shows how all features are combined
4. **Attention Weights:** Shows how attention selects important time steps

---

## References

- **TCN Architecture:** Based on Temporal Convolutional Networks (Bai et al., 2018)
- **Dilated RNN:** Based on Dilated Recurrent Neural Networks (Chang et al., 2017)
- **Implementation:** `src/vitaldb_aki/models/architectures.py::TemporalSynergyClassifier`

---

**Last Updated:** December 2024

