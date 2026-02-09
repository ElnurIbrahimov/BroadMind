# BroadMind v0.74

A neural program executor that learns to run small programs through latent reasoning, wisdom distillation, and adaptive compute.

## What It Does

BroadMind takes a program (a sequence of operations on 3 variables) and an initial state, then predicts the final state. It does this without hardcoded rules -- it learns latent "reasoning circuits" from data.

**4 operation families (13 ops total):**

| Family | Operations | Effect |
|---|---|---|
| ACCUMULATE | ACC_X, ACC_Y, ACC_Z | Increment a variable |
| TRANSFER | TRANSFER_XY, TRANSFER_YZ, TRANSFER_ZX | Move value between variables |
| COMPARE | IF_X_GT_Y_INC_Z, ... | Conditional increment |
| DECREMENT | DEC_X, DEC_Y, DEC_Z | Decrement a variable |

Programs can mix operations from any family.

## Architecture

Three core capabilities built on top of a latent program solver:

1. **Wisdom Distillation** -- compresses many training experiences into a small "wisdom code" per task family. New problems get matched to the right wisdom, guiding the solver.

2. **Adaptive Compute** -- a halter network learns when computation is done. 1-step programs use 1 step; 4-step programs use 4 steps. No wasted compute.

3. **Mixed-Program Execution** -- programs can freely combine ops from all families. The model handles cross-family programs after fine-tuning with mixed batches.

## Training

5-phase curriculum:

| Phase | What | Iterations |
|---|---|---|
| 1 | Solver only (no wisdom) | 2500 |
| 2 | Wisdom distillation | 1500 |
| 2b | Wisdom matcher routing | 800 |
| 3 | Halter only (solver frozen) | 1500 |
| 4 | End-to-end fine-tune (mixed_prob=0.3) | 1500 |

## Results

```
Overall Accuracy:       99.6%
Mixed-Program Accuracy: 99.7%
Wisdom Matching:        100.0%
Adaptive Steps Match:   exact

Per-Family:
  ACCUMULATE: 100%
  TRANSFER:   100%
  COMPARE:    98%
  DECREMENT:  100%

Per-Length:
  Len 1: 100% @ 1.0 steps
  Len 2: 100% @ 2.0 steps
  Len 3:  99% @ 3.0 steps
  Len 4:  99% @ 4.0 steps
```

~477K parameters. Runs on CPU or CUDA.

## Usage

```bash
python FluxMind_v074_complete.py
```

Trains from scratch and saves the checkpoint to `broadmind_v074_complete.pt`.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
