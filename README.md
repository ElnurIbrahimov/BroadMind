# BroadMind

A neural program executor that learns to run programs through latent reasoning, wisdom distillation, adaptive compute, and mixture of recursions. **Not a transformer** -- it generates and executes its own internal programs at runtime.

## Latest: v0.76 (Mixture of Recursions)

| Version | What It Adds | Params | Accuracy |
|---|---|---|---|
| v0.74 | Wisdom distillation + Adaptive compute | 477K | 99.7% |
| v0.75 | Length generalization (sinusoidal encoding, noise injection) | ~421K | 99%+ ID, tested to 16 steps |
| v0.76 | Mixture of Recursions (adaptive inner depth per step) | ~445K | 99%+ with depth hierarchy |

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

Five core capabilities:

1. **Latent Program Induction** -- the model generates its own internal instructions (96-dim latent vectors) at runtime, conditioned on the current state. This is runtime program synthesis, not pattern matching.

2. **Wisdom Distillation** -- compresses many training experiences into a small "wisdom code" (48 floats) per task family. New problems get matched to the right wisdom, guiding the solver.

3. **Adaptive Compute** -- a halter network learns when computation is done. 1-step programs use 1 step; 4-step programs use 4 steps. No wasted compute.

4. **Length Generalization** (v0.75) -- sinusoidal step encodings replace learned embeddings, enabling generalization to program lengths never seen during training. Gaussian noise injection during training improves robustness.

5. **Mixture of Recursions** (v0.76) -- within each solver step, shared weights can iterate 1-4 times. A learned router selects depth based on operation complexity. Simple ops (ACC, DEC) exit early; complex ops (COMPARE) get deeper processing.

```
Input: initial_state (x, y, z) + program [op1, op2, ..., opN]
                |
    [WisdomMatcher] -> select wisdom code from bank
                |
    For each step t:
        [LatentGenerator(state, op, step, wisdom)] -> latent instruction
        [RecursionRouter] -> select inner depth (1-4)  (v0.76)
        For r in range(depth):
            [LatentExecutor(state, latent, comparisons)] -> delta
            state = state + delta
        [Halter] -> should we stop?
                |
    Output: predicted final state
```

## Training

6-phase curriculum (v0.76):

| Phase | What | Iterations |
|---|---|---|
| 1 | Solver only (no wisdom) | 2500 |
| 2 | Wisdom distillation | 1500 |
| 2b | Wisdom matcher routing | 800 |
| 3 | Halter only (solver frozen) | 1500 |
| 4 | End-to-end fine-tune (mixed_prob=0.3) | 1500 |
| 5 | Length generalization (noise injection) | 500 |

## Results (v0.76)

```
Overall Accuracy:       99%+
Mixed-Program Accuracy: 99%+
Wisdom Matching:        100%
Adaptive Steps Match:   exact

Per-Family:
  ACCUMULATE: 100%
  TRANSFER:   100%
  COMPARE:    98%+
  DECREMENT:  100%

Length Generalization (trained on 1-4):
  Length  4: 99%+
  Length  8: 90%+
  Length 12: 85%+
  Length 16: 80%+

MoR Depth Hierarchy:
  COMPARE ops get deeper recursion than TRANSFER > ACC/DEC
```

~445K parameters. Runs on CPU or CUDA.

## Usage

```bash
# Latest (v0.76 - Mixture of Recursions)
python BroadMind_v076_mor.py

# v0.75 (Length Generalization)
python BroadMind_v075_scaling.py

# v0.74 (Complete base model)
python BroadMind_v074_complete.py
```

Each script trains from scratch and saves a checkpoint.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan:

- **v0.75** -- Task Scaling (length generalization) -- DONE
- **v0.76** -- Mixture of Recursions (adaptive inner depth) -- DONE
- **v0.77** -- Hardware-Adaptive Compute: one model, many deployment profiles
- **v0.78** -- Edge Deployment: run on Raspberry Pi, phones, microcontrollers

## Requirements

- Python 3.8+
- PyTorch
- NumPy
