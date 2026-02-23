"""
BroadMind v0.79d: Meta-Learning Wisdom (FluxMind Integration)
==============================================================
Proprietary - Elnur Ibrahimov
February 2026

v0.79d key insight: the solver ALWAYS uses wisdom_to_op(wisdom) for op encoding.
No op_embedding at inference. The solver is wisdom-dependent BY CONSTRUCTION.
Combined with procedural op diversity during training, this enables true
meta-learning generalization to novel operations.

Architecture:
- MetaWisdomEncoder generates per-OP 48D wisdom from K=16 demo transitions
- Per-step wisdom: each program step gets its own wisdom from its op's demos
- wisdom_to_op(wisdom) replaces op_embedding everywhere in the solver
- ProceduralOp generator provides training diversity (50+ random op types)

Loads solver + halter from v0.77 checkpoint, trains MetaWisdomEncoder on top.
Self-contained: run on RunPod with RTX 5090 (~45 min).
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random
import os
import json
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    n_variables = 3
    n_train_families = 4
    n_novel_families = 6
    max_program_length = 4
    max_eval_length = 16
    value_range = 10
    comparison_scale = 30.0

    # Architecture (same as v0.77 solver)
    d_model = 192
    d_latent = 96
    d_wisdom = 48
    d_demo = 96           # demo embedding dim

    # Meta-wisdom
    n_support = 16        # K=16 support transitions per op
    n_cross_heads = 4     # cross-attention heads
    cross_head_dim = 24   # head_dim for cross-attention (4 * 24 = 96 = d_demo)

    # Halting
    halt_threshold = 0.5

    # Training iterations per phase
    n_iterations_phase1 = 2500   # MetaWisdom only, 4 named families, per-op wisdom
    n_iterations_phase2 = 2500   # Joint solver+MetaWisdom, 4 named families
    n_iterations_phase3 = 3000   # Diversity: 50% named + 50% procedural ops
    n_iterations_phase4 = 800    # Halter only
    n_iterations_phase5 = 1500   # End-to-end (named + procedural)
    n_iterations_phase6 = 1000   # Length generalization + noise

    # Procedural ops
    procedural_prob = 0.5        # fraction of Phase 3/5 batches using procedural ops

    # Training hyperparams
    batch_size = 256
    lr = 1e-3
    lr_joint = 5e-4
    lr_fine = 1e-4
    grad_clip = 1.0
    dropout = 0.1
    weight_decay = 1e-4
    noise_std = 0.05

    # MoR (unchanged from v0.77)
    max_recursion_depth = 4
    recursion_enc_dim = 24
    compute_cost_weight = 0.005

    # Wisdom anchoring (Phase 1)
    anchor_weight = 2.0    # stronger anchoring
    anchor_decay = 0.999   # slower decay — keep anchoring longer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ============================================================================
# TRAINING TASK FAMILIES (same 4 as v0.77)
# ============================================================================

TRAIN_FAMILIES = {
    0: {'name': 'ACCUMULATE', 'ops': ['ACC_X', 'ACC_Y', 'ACC_Z']},
    1: {'name': 'TRANSFER', 'ops': ['TRANSFER_XY', 'TRANSFER_YZ', 'TRANSFER_ZX']},
    2: {'name': 'COMPARE', 'ops': ['IF_X_GT_Y_INC_Z', 'IF_Y_GT_Z_INC_X', 'IF_Z_GT_X_INC_Y']},
    3: {'name': 'DECREMENT', 'ops': ['DEC_X', 'DEC_Y', 'DEC_Z']},
}

# ============================================================================
# NOVEL TASK FAMILIES (6 held-out families for meta-learning eval)
# ============================================================================

NOVEL_FAMILIES = {
    4: {'name': 'SWAP', 'difficulty': 'easy',
        'ops': ['SWAP_XY', 'SWAP_YZ', 'SWAP_ZX']},
    5: {'name': 'MIRROR', 'difficulty': 'easy',
        'ops': ['MIRROR_X', 'MIRROR_Y', 'MIRROR_Z']},
    6: {'name': 'DOUBLE', 'difficulty': 'medium',
        'ops': ['DOUBLE_X', 'DOUBLE_Y', 'DOUBLE_Z']},
    7: {'name': 'MIN_MAX', 'difficulty': 'medium',
        'ops': ['MIN_XY_Z', 'MIN_YZ_X', 'MIN_ZX_Y']},
    8: {'name': 'MODULAR', 'difficulty': 'hard',
        'ops': ['MOD_XY', 'MOD_YZ', 'MOD_ZX']},
    9: {'name': 'COND_TRANSFER', 'difficulty': 'hard',
        'ops': ['COND_X_YZ', 'COND_Y_ZX', 'COND_Z_XY']},
}

ALL_FAMILIES = {**TRAIN_FAMILIES, **NOVEL_FAMILIES}

# Build op lists — training ops + NOVEL token
TRAIN_OPS = []
for fam in TRAIN_FAMILIES.values():
    TRAIN_OPS.extend(fam['ops'])
TRAIN_OPS.append('PAD')
TRAIN_OPS.append('NOVEL')   # special token for op-agnostic training

PAD_IDX = TRAIN_OPS.index('PAD')
NOVEL_IDX = TRAIN_OPS.index('NOVEL')
N_OPS = len(TRAIN_OPS)
OP_TO_IDX = {op: i for i, op in enumerate(TRAIN_OPS)}
IDX_TO_OP = {i: op for op, i in OP_TO_IDX.items()}

# Novel ops get separate list (not in the embedding table)
NOVEL_OPS = []
for fam in NOVEL_FAMILIES.values():
    NOVEL_OPS.extend(fam['ops'])

def get_family_id(op_name):
    for fam_id, fam in ALL_FAMILIES.items():
        if op_name in fam['ops']:
            return fam_id
    return -1

# ============================================================================
# PROCEDURAL OPERATIONS (random parameterized ops for training diversity)
# ============================================================================

class ProceduralOp:
    """A randomly parameterized operation on 3-variable states.

    Provides training diversity so MetaWisdomEncoder learns general
    transformation semantics instead of classifying into 4 buckets.
    18 op types x 3 sources x 3 targets x params = 500+ unique ops.
    """
    OP_TYPES = [
        'add_const', 'sub_const', 'set_const',
        'copy', 'add_vars', 'sub_vars',
        'min_vars', 'max_vars',
        'negate', 'double', 'halve',
        'clamp_high', 'clamp_low',
        'cond_inc', 'cond_dec',
        'swap', 'mirror', 'modular',
    ]

    def __init__(self):
        self.op_type = random.choice(self.OP_TYPES)
        self.source = random.randint(0, 2)
        self.target = random.randint(0, 2)
        self.source2 = random.randint(0, 2)
        self.param = random.randint(1, 5)
        self.threshold = random.randint(3, 7)
        self._id = id(self)

    def execute(self, state):
        vals = list(state)
        src = vals[self.source]
        src2 = vals[self.source2]
        vr = config.value_range

        if self.op_type == 'add_const':
            vals[self.target] = min(src + self.param, vr)
        elif self.op_type == 'sub_const':
            vals[self.target] = max(src - self.param, 0)
        elif self.op_type == 'set_const':
            vals[self.target] = min(self.param, vr)
        elif self.op_type == 'copy':
            vals[self.target] = src
        elif self.op_type == 'add_vars':
            vals[self.target] = min(src + src2, vr)
        elif self.op_type == 'sub_vars':
            vals[self.target] = max(src - src2, 0)
        elif self.op_type == 'min_vars':
            vals[self.target] = min(src, src2)
        elif self.op_type == 'max_vars':
            vals[self.target] = max(src, src2)
        elif self.op_type == 'negate':
            vals[self.target] = vr - src
        elif self.op_type == 'double':
            vals[self.target] = min(2 * src, vr)
        elif self.op_type == 'halve':
            vals[self.target] = src // 2
        elif self.op_type == 'clamp_high':
            vals[self.target] = min(src, self.param)
        elif self.op_type == 'clamp_low':
            vals[self.target] = max(src, self.param)
        elif self.op_type == 'cond_inc':
            if src > self.threshold:
                vals[self.target] = min(vals[self.target] + 1, vr)
        elif self.op_type == 'cond_dec':
            if src > self.threshold:
                vals[self.target] = max(vals[self.target] - 1, 0)
        elif self.op_type == 'swap':
            vals[self.source], vals[self.target] = vals[self.target], vals[self.source]
        elif self.op_type == 'mirror':
            vals[self.target] = vr - src
        elif self.op_type == 'modular':
            vals[self.target] = src % max(src2, 1)
        return tuple(vals)


def generate_procedural_family(n_ops=3):
    """Generate a family of n_ops random procedural operations."""
    return [ProceduralOp() for _ in range(n_ops)]

# ============================================================================
# OPERATIONS (training + novel)
# ============================================================================

def execute_op(state, op_name):
    x, y, z = state
    # --- Training ops (same as v0.77) ---
    if op_name == 'ACC_X': return (x + 1, y, z)
    if op_name == 'ACC_Y': return (x, y + 1, z)
    if op_name == 'ACC_Z': return (x, y, z + 1)
    if op_name == 'TRANSFER_XY': return (x - 1, y + 1, z)
    if op_name == 'TRANSFER_YZ': return (x, y - 1, z + 1)
    if op_name == 'TRANSFER_ZX': return (x + 1, y, z - 1)
    if op_name == 'IF_X_GT_Y_INC_Z': return (x, y, z + 1) if x > y else (x, y, z)
    if op_name == 'IF_Y_GT_Z_INC_X': return (x + 1, y, z) if y > z else (x, y, z)
    if op_name == 'IF_Z_GT_X_INC_Y': return (x, y + 1, z) if z > x else (x, y, z)
    if op_name == 'DEC_X': return (x - 1, y, z)
    if op_name == 'DEC_Y': return (x, y - 1, z)
    if op_name == 'DEC_Z': return (x, y, z - 1)
    if op_name == 'PAD': return (x, y, z)

    # --- Novel ops ---
    # SWAP (easy)
    if op_name == 'SWAP_XY': return (y, x, z)
    if op_name == 'SWAP_YZ': return (x, z, y)
    if op_name == 'SWAP_ZX': return (z, y, x)
    # MIRROR (easy)
    if op_name == 'MIRROR_X': return (config.value_range - x, y, z)
    if op_name == 'MIRROR_Y': return (x, config.value_range - y, z)
    if op_name == 'MIRROR_Z': return (x, y, config.value_range - z)
    # DOUBLE (medium)
    if op_name == 'DOUBLE_X': return (min(2 * x, config.value_range), y, z)
    if op_name == 'DOUBLE_Y': return (x, min(2 * y, config.value_range), z)
    if op_name == 'DOUBLE_Z': return (x, y, min(2 * z, config.value_range))
    # MIN_MAX (medium)
    if op_name == 'MIN_XY_Z': return (x, y, min(x, y))
    if op_name == 'MIN_YZ_X': return (min(y, z), y, z)
    if op_name == 'MIN_ZX_Y': return (x, min(z, x), z)
    # MODULAR (hard)
    if op_name == 'MOD_XY': return (x % max(y, 1), y, z)
    if op_name == 'MOD_YZ': return (x, y % max(z, 1), z)
    if op_name == 'MOD_ZX': return (x, y, z % max(x, 1))
    # COND_TRANSFER (hard)
    if op_name == 'COND_X_YZ': return (x, y - 1, z + 1) if x > 5 else (x, y, z)
    if op_name == 'COND_Y_ZX': return (x + 1, y, z - 1) if y > 5 else (x, y, z)
    if op_name == 'COND_Z_XY': return (x - 1, y + 1, z) if z > 5 else (x, y, z)

    raise ValueError(f"Unknown op: {op_name}")

def execute_program(initial_state, program_ops):
    states = [initial_state]
    state = initial_state
    for op_name in program_ops:
        state = execute_op(state, op_name)
        states.append(state)
    return states

# ============================================================================
# SINUSOIDAL STEP ENCODING (unchanged from v0.77)
# ============================================================================

def sinusoidal_step_encoding(step_num, d, batch_size, device):
    position = torch.tensor([step_num], dtype=torch.float, device=device)
    div_term = torch.exp(
        torch.arange(0, d, 2, device=device).float() * -(math.log(10000.0) / d)
    )
    pe = torch.zeros(1, d, device=device)
    pe[0, 0::2] = torch.sin(position * div_term)
    pe[0, 1::2] = torch.cos(position * div_term)
    return pe.expand(batch_size, -1)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_batch(batch_size, min_len=1, max_len=4, families=None, mixed_prob=0.0):
    """Generate training batch with per-step op tracking.

    Returns op_names: list of lists of op name strings (for per-op wisdom).
    For training families: op indices from OP_TO_IDX (for halter).
    For novel families: op index = NOVEL_IDX (halter just counts steps).
    """
    if families is None:
        families = TRAIN_FAMILIES

    program_indices = []
    initial_states = []
    all_intermediate = []
    lengths = []
    family_ids = []
    op_names_batch = []

    all_family_ops = []
    for fam in families.values():
        all_family_ops.extend(fam['ops'])

    is_novel = any(fam_id >= 4 for fam_id in families.keys())

    for _ in range(batch_size):
        prog_len = random.randint(min_len, max_len)

        if random.random() < mixed_prob and not is_novel:
            prog_ops = [random.choice(all_family_ops) for _ in range(prog_len)]
            family_id = get_family_id(prog_ops[0])
        else:
            family_id = random.choice(list(families.keys()))
            ops = families[family_id]['ops']
            prog_ops = [random.choice(ops) for _ in range(prog_len)]

        init = tuple(np.random.randint(0, config.value_range) for _ in range(3))

        # Op indices for halter (solver uses wisdom instead)
        prog_idx = []
        for op in prog_ops:
            if op in OP_TO_IDX:
                prog_idx.append(OP_TO_IDX[op])
            else:
                prog_idx.append(NOVEL_IDX)

        states = execute_program(init, prog_ops)

        # Pad ops and op_names
        full_ops = list(prog_ops)
        while len(prog_idx) < max_len:
            prog_idx.append(PAD_IDX)
        while len(full_ops) < max_len:
            full_ops.append('PAD')

        intermediate = list(states[1:])
        while len(intermediate) < max_len:
            intermediate.append(intermediate[-1])

        program_indices.append(prog_idx)
        initial_states.append(init)
        all_intermediate.append(intermediate)
        lengths.append(prog_len)
        family_ids.append(family_id)
        op_names_batch.append(full_ops)

    return {
        'program_indices': torch.tensor(program_indices, dtype=torch.long, device=config.device),
        'initial_states': torch.tensor(initial_states, dtype=torch.float, device=config.device),
        'intermediate': torch.tensor(all_intermediate, dtype=torch.float, device=config.device),
        'lengths': torch.tensor(lengths, dtype=torch.long, device=config.device),
        'family_ids': torch.tensor(family_ids, dtype=torch.long, device=config.device),
        'op_names': op_names_batch,
    }


def generate_support_transitions(families, n_support=16):
    """Generate K support (state, next_state) transitions for meta-wisdom.

    Returns:
        states:      (n_support, 3) — before states
        next_states: (n_support, 3) — after states
    """
    states_list = []
    next_list = []

    all_ops = []
    for fam in families.values():
        all_ops.extend(fam['ops'])

    for _ in range(n_support):
        op = random.choice(all_ops)
        init = tuple(np.random.randint(0, config.value_range) for _ in range(3))
        result = execute_op(init, op)
        states_list.append(init)
        next_list.append(result)

    return (
        torch.tensor(states_list, dtype=torch.float, device=config.device),
        torch.tensor(next_list, dtype=torch.float, device=config.device),
    )


def generate_support_batch(families, batch_size, n_support=16):
    """Generate a batch of support sets (one per sample).

    Returns:
        states:      (batch_size, n_support, 3)
        next_states: (batch_size, n_support, 3)
    """
    all_states = []
    all_next = []
    for _ in range(batch_size):
        s, ns = generate_support_transitions(families, n_support)
        all_states.append(s)
        all_next.append(ns)
    return torch.stack(all_states), torch.stack(all_next)


def generate_per_op_support(op_name, n_support=16):
    """Generate K demo transitions for one specific named op."""
    states_list, next_list = [], []
    for _ in range(n_support):
        init = tuple(np.random.randint(0, config.value_range) for _ in range(3))
        result = execute_op(init, op_name)
        states_list.append(init)
        next_list.append(result)
    return (
        torch.tensor(states_list, dtype=torch.float, device=config.device),
        torch.tensor(next_list, dtype=torch.float, device=config.device),
    )


def generate_procedural_support(proc_op, n_support=16):
    """Generate K demo transitions for one procedural op."""
    states_list, next_list = [], []
    for _ in range(n_support):
        init = tuple(np.random.randint(0, config.value_range) for _ in range(3))
        result = proc_op.execute(init)
        states_list.append(init)
        next_list.append(result)
    return (
        torch.tensor(states_list, dtype=torch.float, device=config.device),
        torch.tensor(next_list, dtype=torch.float, device=config.device),
    )


def generate_procedural_batch(batch_size, min_len=1, max_len=4, n_ops=3):
    """Generate batch using random procedural operations.

    All ops use NOVEL_IDX for halter. op_names are ProceduralOp objects.
    A fresh procedural family (n_ops random ops) is created per call.
    """
    proc_ops = generate_procedural_family(n_ops)

    program_indices = []
    initial_states = []
    all_intermediate = []
    lengths = []
    op_names_batch = []

    for _ in range(batch_size):
        prog_len = random.randint(min_len, max_len)
        step_ops = [random.choice(proc_ops) for _ in range(prog_len)]

        init = tuple(np.random.randint(0, config.value_range) for _ in range(3))

        # Execute procedural program
        state = init
        states = []
        for op in step_ops:
            state = op.execute(state)
            states.append(state)

        # All ops are NOVEL for halter
        prog_idx = [NOVEL_IDX] * prog_len
        while len(prog_idx) < max_len:
            prog_idx.append(PAD_IDX)

        intermediate = list(states)
        while len(intermediate) < max_len:
            intermediate.append(intermediate[-1])

        # Pad op_names with last op (for padded steps)
        full_ops = list(step_ops)
        while len(full_ops) < max_len:
            full_ops.append(step_ops[-1])

        program_indices.append(prog_idx)
        initial_states.append(init)
        all_intermediate.append(intermediate)
        lengths.append(prog_len)
        op_names_batch.append(full_ops)

    return {
        'program_indices': torch.tensor(program_indices, dtype=torch.long, device=config.device),
        'initial_states': torch.tensor(initial_states, dtype=torch.float, device=config.device),
        'intermediate': torch.tensor(all_intermediate, dtype=torch.float, device=config.device),
        'lengths': torch.tensor(lengths, dtype=torch.long, device=config.device),
        'op_names': op_names_batch,
    }


def compute_batch_wisdom(model, op_names_batch, max_steps, aggregate_only=True):
    """Compute per-step wisdom for a batch by generating per-op support.

    Efficiently batches all unique ops into one MetaWisdomEncoder forward pass.

    Args:
        op_names_batch: list of lists of (str or ProceduralOp)
        max_steps: number of steps per program
        aggregate_only: use mean-pool (True) or cross-attention (False)

    Returns: (batch_size, max_steps, d_wisdom)
    """
    batch_size = len(op_names_batch)

    # Collect unique ops
    unique_ops = {}
    for ops in op_names_batch:
        for op in ops:
            key = op if isinstance(op, str) else op._id
            if key not in unique_ops:
                unique_ops[key] = op

    # Generate support for each unique op
    all_demo_s, all_demo_ns = [], []
    key_order = list(unique_ops.keys())

    for key in key_order:
        op = unique_ops[key]
        if isinstance(op, str):
            s, ns = generate_per_op_support(op, config.n_support)
        else:
            s, ns = generate_procedural_support(op, config.n_support)
        all_demo_s.append(s)
        all_demo_ns.append(ns)

    all_demo_s = torch.stack(all_demo_s)     # (n_unique, K, 3)
    all_demo_ns = torch.stack(all_demo_ns)   # (n_unique, K, 3)
    dummy_states = torch.zeros(len(key_order), 3, device=config.device)

    # One forward pass for all unique ops
    all_wisdoms = model.meta_wisdom(
        dummy_states, all_demo_s, all_demo_ns, aggregate_only=aggregate_only
    )  # (n_unique, d_wisdom)

    # Assign per-step wisdom
    key_to_idx = {k: i for i, k in enumerate(key_order)}
    wisdoms = torch.zeros(batch_size, max_steps, config.d_wisdom, device=config.device)

    for b in range(batch_size):
        for t in range(min(len(op_names_batch[b]), max_steps)):
            op = op_names_batch[b][t]
            key = op if isinstance(op, str) else op._id
            wisdoms[b, t] = all_wisdoms[key_to_idx[key]]

    return wisdoms

# ============================================================================
# SWITCHABLE LAYER NORM (from v0.77)
# ============================================================================

class SwitchableLayerNorm(nn.Module):
    def __init__(self, full_dim, width_multipliers):
        super().__init__()
        self.full_dim = full_dim
        self.norms = nn.ModuleDict()
        for w in width_multipliers:
            key = f"w{int(w * 100)}"
            dim = int(full_dim * w)
            self.norms[key] = nn.LayerNorm(dim)

    def forward(self, x, width_mult=1.0):
        key = f"w{int(width_mult * 100)}"
        return self.norms[key](x)

# ============================================================================
# META-WISDOM ENCODER (~95K params — replaces WisdomBank+Distiller+Matcher)
# ============================================================================

class MetaWisdomEncoder(nn.Module):
    """Generates 48D wisdom from K demonstration transitions using cross-attention.

    Architecture:
        TransitionEncoder: encode each (state, next_state) pair → d_demo (96D)
        DeltaEncoder: encode (next_state - state) → d_demo (96D)
        Combiner: fuse transition + delta → d_demo (96D) per demo
        CrossAttention: query=problem_context, keys/values=demo embeddings
        WisdomHead: LayerNorm → Linear → Tanh → d_wisdom (48D)

    Op-agnostic: only sees (state, next_state) pairs and deltas. No op identity.
    """

    def __init__(self, config):
        super().__init__()
        d = config.d_demo  # 96

        # Encode (state, next_state) concatenated → 6 dims input
        self.transition_encoder = nn.Sequential(
            nn.Linear(config.n_variables * 2, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
        )

        # Encode delta (next_state - state) → 3 dims input
        self.delta_encoder = nn.Sequential(
            nn.Linear(config.n_variables, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Combine transition + delta → d_demo
        self.combiner = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
        )

        # Problem context encoder (state → query for cross-attention)
        self.context_encoder = nn.Sequential(
            nn.Linear(config.n_variables, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Cross-attention: query=context, key/value=demo embeddings
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.n_cross_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Wisdom head: project cross-attention output → d_wisdom
        self.wisdom_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, config.d_wisdom),
            nn.Tanh(),
        )

    def encode_demos(self, demo_states, demo_next_states):
        """Encode demonstration transitions.

        Args:
            demo_states:      (batch, K, 3) or (K, 3)
            demo_next_states: (batch, K, 3) or (K, 3)

        Returns:
            demo_emb: (batch, K, d_demo) or (K, d_demo)
        """
        squeeze = False
        if demo_states.dim() == 2:
            squeeze = True
            demo_states = demo_states.unsqueeze(0)
            demo_next_states = demo_next_states.unsqueeze(0)

        batch, K, _ = demo_states.shape

        # Flatten for encoding
        s_flat = demo_states.reshape(batch * K, -1)
        ns_flat = demo_next_states.reshape(batch * K, -1)

        # Transition: concat (state, next_state)
        trans_input = torch.cat([s_flat, ns_flat], dim=-1)
        trans_emb = self.transition_encoder(trans_input)

        # Delta
        delta = ns_flat - s_flat
        delta_emb = self.delta_encoder(delta)

        # Combine
        combined = torch.cat([trans_emb, delta_emb], dim=-1)
        demo_emb = self.combiner(combined)

        demo_emb = demo_emb.reshape(batch, K, -1)
        if squeeze:
            demo_emb = demo_emb.squeeze(0)
        return demo_emb

    def forward(self, problem_states, demo_states, demo_next_states,
                demo_emb=None, aggregate_only=False):
        """Generate wisdom from demonstrations for given problem states.

        Args:
            problem_states:   (batch, 3)
            demo_states:      (batch, K, 3) or (K, 3)
            demo_next_states: (batch, K, 3) or (K, 3)
            demo_emb:         (batch, K, d_demo) pre-encoded (optional)
            aggregate_only:   if True, mean-pool demos (no cross-attention on
                              problem state). Gives consistent per-family wisdom.

        Returns:
            wisdom: (batch, d_wisdom) — 48D wisdom vector
        """
        batch_size = problem_states.shape[0]

        # Encode demos if not pre-encoded
        if demo_emb is None:
            if demo_states.dim() == 2:
                demo_emb = self.encode_demos(demo_states, demo_next_states)
                demo_emb = demo_emb.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                demo_emb = self.encode_demos(demo_states, demo_next_states)
        elif demo_emb.dim() == 2:
            demo_emb = demo_emb.unsqueeze(0).expand(batch_size, -1, -1)

        if aggregate_only:
            # Mean-pool demos — no state dependency, consistent per-family wisdom
            pooled = demo_emb.mean(dim=1)                # (batch, d_demo)
            wisdom = self.wisdom_head(pooled)             # (batch, d_wisdom)
        else:
            # Cross-attention with problem state as query
            context = self.context_encoder(problem_states)
            query = context.unsqueeze(1)
            attended, _ = self.cross_attention(query, demo_emb, demo_emb)
            attended = attended.squeeze(1)
            wisdom = self.wisdom_head(attended)

        return wisdom

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# ELASTIC SOLVER (from v0.77 — unchanged, just needs compatible N_OPS)
# ============================================================================

class ElasticSolver(nn.Module):
    """Wisdom-guided program executor with MoR and elastic width/depth.

    v0.79d: solver ALWAYS uses wisdom_to_op(wisdom) for op encoding.
    op_embedding is kept for initialization but NOT used during forward pass.
    This makes the solver wisdom-dependent by construction — it MUST learn
    to interpret wisdom-derived features for any operation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        width_multipliers = [1.0]

        # State encoder
        self.state_enc_linear = nn.Linear(config.n_variables, d)
        self.state_enc_norm = SwitchableLayerNorm(d, width_multipliers)

        # Op embedding (includes NOVEL token)
        self.op_embedding = nn.Embedding(N_OPS, d)

        # Wisdom-to-op projection: maps 48D wisdom → 192D op-embedding space
        # Used instead of op_embedding when op is NOVEL
        self.wisdom_to_op = nn.Linear(config.d_wisdom, d)

        # Wisdom encoder (for latent gen — separate from wisdom_to_op)
        self.wisdom_enc_linear = nn.Linear(config.d_wisdom, d // 2)

        # Recursion router
        self.router_linear1 = nn.Linear(d * 2, d // 4)
        self.router_linear2 = nn.Linear(d // 4, config.max_recursion_depth)
        with torch.no_grad():
            self.router_linear2.bias.data = torch.tensor([2.0, 0.0, -1.0, -2.0])

        # Latent generator
        latent_gen_in = d * 2 + d // 4 + d // 2 + config.recursion_enc_dim
        self.latent_gen_linear1 = nn.Linear(latent_gen_in, d)
        self.latent_gen_norm = SwitchableLayerNorm(d, width_multipliers)
        self.latent_gen_linear2 = nn.Linear(d, config.d_latent)

        # Comparison encoder
        self.comp_enc_linear = nn.Linear(6, d // 4)

        # Latent encoder
        self.latent_enc_linear = nn.Linear(config.d_latent, d)
        self.latent_enc_norm = SwitchableLayerNorm(d, width_multipliers)

        # Executor
        executor_in = d * 2 + d // 4
        self.exec_linear1 = nn.Linear(executor_in, d)
        self.exec_norm = SwitchableLayerNorm(d, width_multipliers)
        self.exec_linear2 = nn.Linear(d, d)
        self.exec_linear3 = nn.Linear(d, config.n_variables)

        # Predictor
        self.predictor = nn.Linear(config.n_variables, config.n_variables)

    def _encode_state(self, state):
        enc = self.state_enc_linear(state)
        enc = self.state_enc_norm(enc, 1.0)
        enc = F.gelu(enc)
        enc = F.dropout(enc, p=self.config.dropout, training=self.training)
        return enc

    def _comparison_features(self, state):
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        return torch.stack([
            (x > y).float(), (y > z).float(), (z > x).float(),
            (x - y) / self.config.comparison_scale,
            (y - z) / self.config.comparison_scale,
            (z - x) / self.config.comparison_scale,
        ], dim=-1)

    def step(self, state, op_idx, step_num, wisdom, training_noise_std=0.0):
        batch_size = state.shape[0]
        d = self.config.d_model

        # ALWAYS use wisdom_to_op — solver is wisdom-dependent by construction
        # op_embedding is NOT used here (kept for initialization/anchoring only)
        op_enc = self.wisdom_to_op(wisdom)  # (batch, d_model)

        step_enc = sinusoidal_step_encoding(step_num, d // 4, batch_size, state.device)
        wisdom_enc = F.gelu(self.wisdom_enc_linear(wisdom))

        # Router
        state_enc = self._encode_state(state)
        router_input = torch.cat([state_enc, op_enc], dim=-1)
        r1 = F.gelu(self.router_linear1(router_input))
        router_logits = self.router_linear2(r1)

        if self.training:
            router_weights = F.gumbel_softmax(router_logits, tau=1.0, hard=True)
        else:
            router_weights = F.one_hot(
                router_logits.argmax(dim=-1), self.config.max_recursion_depth
            ).float()

        # Inner recursion
        recursive_state = state
        outputs = []

        for r in range(self.config.max_recursion_depth):
            state_enc_r = self._encode_state(recursive_state)
            rec_enc = sinusoidal_step_encoding(
                r, self.config.recursion_enc_dim, batch_size, state.device
            )

            latent_input = torch.cat(
                [state_enc_r, op_enc, step_enc, wisdom_enc, rec_enc], dim=-1
            )
            lg1 = self.latent_gen_linear1(latent_input)
            lg1 = self.latent_gen_norm(lg1, 1.0)
            lg1 = F.gelu(lg1)
            lg1 = F.dropout(lg1, p=self.config.dropout, training=self.training)
            latent = self.latent_gen_linear2(lg1)

            comp_features = self._comparison_features(recursive_state)
            comp_enc = F.gelu(self.comp_enc_linear(comp_features))

            latent_enc = self.latent_enc_linear(latent)
            latent_enc = self.latent_enc_norm(latent_enc, 1.0)
            latent_enc = F.gelu(latent_enc)

            exec_input = torch.cat([state_enc_r, latent_enc, comp_enc], dim=-1)
            e1 = self.exec_linear1(exec_input)
            e1 = self.exec_norm(e1, 1.0)
            e1 = F.gelu(e1)
            e1 = F.dropout(e1, p=self.config.dropout, training=self.training)
            e2 = F.gelu(self.exec_linear2(e1))
            delta = self.exec_linear3(e2)

            recursive_state = recursive_state + delta
            outputs.append(recursive_state)

        stacked = torch.stack(outputs, dim=1)
        new_state = (stacked * router_weights.unsqueeze(-1)).sum(dim=1)

        depths = torch.tensor([1.0, 2.0, 3.0, 4.0], device=state.device)
        compute_cost = (router_weights * depths).sum(-1).mean()

        if training_noise_std > 0.0 and self.training:
            new_state = new_state + torch.randn_like(new_state) * training_noise_std

        return new_state, compute_cost

    def forward(self, programs, initial_states, per_step_wisdom, training_noise_std=0.0):
        """Forward with per-step wisdom.

        Args:
            per_step_wisdom: (batch, max_steps, d_wisdom) — each step gets its own wisdom
        """
        n_steps = programs.shape[1]
        state = initial_states
        all_preds = []
        all_states = []
        total_compute_cost = 0.0

        for t in range(n_steps):
            op_idx = programs[:, t]
            wisdom_t = per_step_wisdom[:, t, :]  # (batch, d_wisdom)
            state, compute_cost = self.step(
                state, op_idx, t, wisdom_t,
                training_noise_std=training_noise_std,
            )
            pred = self.predictor(state)
            all_preds.append(pred)
            all_states.append(state)
            total_compute_cost = total_compute_cost + compute_cost

        return torch.stack(all_preds, dim=1), torch.stack(all_states, dim=1), total_compute_cost

# ============================================================================
# HALTER (unchanged from v0.77)
# ============================================================================

class Halter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.op_embedding = nn.Embedding(N_OPS, config.d_model // 2)
        self.program_encoder = nn.Sequential(
            nn.Linear(config.d_model // 2 + 1, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model + config.d_model // 2, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, programs, step):
        batch_size = programs.shape[0]
        op_emb = self.op_embedding(programs)
        pad_mask = (programs != PAD_IDX).float().unsqueeze(-1)
        op_count = pad_mask.sum(dim=1)
        masked_emb = op_emb * pad_mask
        pooled = masked_emb.sum(dim=1) / op_count.clamp(min=1)
        op_count_norm = op_count / config.max_eval_length
        program_input = torch.cat([pooled, op_count_norm], dim=-1)
        program_enc = self.program_encoder(program_input)
        step_enc = sinusoidal_step_encoding(
            step, config.d_model // 2, batch_size, programs.device
        )
        x = torch.cat([program_enc, step_enc], dim=-1)
        logit = self.classifier(x)
        return logit

# ============================================================================
# COMPLETE MODEL
# ============================================================================

class BroadMindV079(nn.Module):
    """BroadMind v0.79: Meta-Learning Wisdom.

    Replaces static wisdom with MetaWisdomEncoder that generates 48D wisdom
    from K demonstration transitions at inference time.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # New: MetaWisdomEncoder replaces WisdomBank + Distiller + Matcher
        self.meta_wisdom = MetaWisdomEncoder(config)

        # Solver (from v0.77, loads pretrained weights)
        self.solver = ElasticSolver(config)

        # Halter (from v0.77, loads pretrained weights)
        self.halter = Halter(config)

    def forward_all_steps(self, programs, initial_states, per_step_wisdom,
                          training_noise_std=0.0):
        """Run all steps with per-step wisdom.

        Args:
            per_step_wisdom: (batch, max_steps, d_wisdom)
        """
        preds, states, compute_cost = self.solver(
            programs, initial_states, per_step_wisdom,
            training_noise_std=training_noise_std,
        )

        halt_logits = []
        for t in range(programs.shape[1]):
            logit = self.halter(programs, t)
            halt_logits.append(logit)

        return preds, torch.stack(halt_logits, dim=1), compute_cost

    def forward_adaptive(self, programs, initial_states, per_step_wisdom,
                         training_noise_std=0.0):
        """Forward with adaptive halting using per-step wisdom."""
        batch_size = programs.shape[0]
        max_steps = programs.shape[1]

        state = initial_states
        halted = torch.zeros(batch_size, dtype=torch.bool, device=state.device)
        final_states = state.clone()
        steps_used = torch.zeros(batch_size, device=state.device)
        total_compute_cost = 0.0

        for t in range(max_steps):
            if halted.all():
                break

            op_idx = programs[:, t]
            wisdom_t = per_step_wisdom[:, t, :]
            new_state, compute_cost = self.solver.step(
                state, op_idx, t, wisdom_t,
                training_noise_std=training_noise_std,
            )
            total_compute_cost = total_compute_cost + compute_cost

            state = torch.where(halted.unsqueeze(1), state, new_state)

            halt_logit = self.halter(programs, t)
            halt_prob = torch.sigmoid(halt_logit).squeeze(-1)

            should_halt = (halt_prob > self.config.halt_threshold) & ~halted

            final_states = torch.where(should_halt.unsqueeze(1), state, final_states)
            steps_used = steps_used + (~halted).float()
            halted = halted | should_halt

        final_states = torch.where(~halted.unsqueeze(1), state, final_states)

        pred = self.solver.predictor(final_states)
        return pred, steps_used, total_compute_cost

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# CHECKPOINT LOADING (v0.77 → v0.79)
# ============================================================================

def load_v077_checkpoint(model, checkpoint_path):
    """Load v0.77 checkpoint into v0.79 model.

    Loads solver + halter weights, skips old wisdom components.
    Returns the old wisdom codes for anchoring loss.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        print("[WARNING] Starting from scratch (no pretrained solver)")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Extract old wisdom codes for anchoring
    old_wisdom_codes = None
    wisdom_key = 'wisdom_bank.wisdom_codes'
    if wisdom_key in state_dict:
        old_wisdom_codes = state_dict[wisdom_key].clone()

    # Map v0.77 keys → v0.79 keys
    new_state_dict = {}
    skipped = []

    for key, value in state_dict.items():
        # Skip old wisdom components
        if any(key.startswith(prefix) for prefix in ['distiller.', 'wisdom_bank.', 'matcher.']):
            skipped.append(key)
            continue

        # Solver weights: direct mapping
        if key.startswith('solver.'):
            new_key = key
            # Handle op_embedding size change (v0.77 had 13 ops, v0.79 has 15 with NOVEL+PAD)
            if 'op_embedding' in key:
                old_n_ops = value.shape[0]
                new_n_ops = N_OPS
                if old_n_ops != new_n_ops:
                    d_emb = value.shape[1]
                    new_emb = torch.randn(new_n_ops, d_emb, device=value.device) * 0.02
                    # Copy existing ops
                    n_copy = min(old_n_ops, new_n_ops)
                    new_emb[:n_copy] = value[:n_copy]
                    value = new_emb
                    print(f"  [RESIZED] {key}: {old_n_ops} -> {new_n_ops} ops")

            # Handle SwitchableLayerNorm — v0.77 has w25/w50/w75/w100, v0.79 only w100
            if '.norms.' in key:
                if 'w100' in key:
                    new_state_dict[new_key] = value
                # Skip sub-width norms
                continue

            new_state_dict[new_key] = value

        # Halter weights: direct mapping (same resize logic for op_embedding)
        elif key.startswith('halter.'):
            new_key = key
            if 'op_embedding' in key:
                old_n_ops = value.shape[0]
                new_n_ops = N_OPS
                if old_n_ops != new_n_ops:
                    d_emb = value.shape[1]
                    new_emb = torch.randn(new_n_ops, d_emb, device=value.device) * 0.02
                    n_copy = min(old_n_ops, new_n_ops)
                    new_emb[:n_copy] = value[:n_copy]
                    value = new_emb
                    print(f"  [RESIZED] {key}: {old_n_ops} -> {new_n_ops} ops")

            new_state_dict[new_key] = value

    # Load into model
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    # Initialize wisdom_to_op: bias = mean of real op embeddings, weights = 0
    # This way NOVEL tokens produce "average op" features before training
    with torch.no_grad():
        real_op_embs = model.solver.op_embedding.weight[:PAD_IDX]  # exclude PAD/NOVEL
        mean_op = real_op_embs.mean(dim=0)
        model.solver.wisdom_to_op.bias.data.copy_(mean_op)
        nn.init.zeros_(model.solver.wisdom_to_op.weight)
    print("  [INIT] wisdom_to_op: bias=mean_op_embedding, weight=zeros")

    # Filter expected missing/unexpected keys
    unexpected_missing = [k for k in missing
                          if not k.startswith('meta_wisdom.')
                          and 'wisdom_to_op' not in k]
    # Junction column buffers from v0.77 elastic width are expected unexpected
    truly_unexpected = [k for k in unexpected if not k.endswith(('_cols_w25', '_cols_w50', '_cols_w75', '_cols_w100'))]

    print(f"[CHECKPOINT] Loaded {len(new_state_dict)} parameters from v0.77")
    print(f"  Skipped wisdom components: {len(skipped)}")
    print(f"  New MetaWisdomEncoder params: {len(missing) - len(unexpected_missing)}")
    if unexpected_missing:
        print(f"  [WARNING] Unexpectedly missing: {unexpected_missing}")
    if truly_unexpected:
        print(f"  [WARNING] Unexpected keys: {truly_unexpected}")

    return old_wisdom_codes

# ============================================================================
# (Old support set helpers removed — replaced by compute_batch_wisdom above)
# ============================================================================

# ============================================================================
# TRAINING PHASES (v0.79d: wisdom-first, procedural diversity)
# ============================================================================

def _compute_task_loss(preds, intermediate, lengths):
    """Compute masked MSE loss over valid program steps."""
    max_steps = preds.shape[1]
    mask = torch.arange(max_steps, device=preds.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()
    return ((preds - intermediate) ** 2 * mask).sum() / mask.sum()


def _compute_anchor_loss(model, batch, old_wisdom_codes, iteration):
    """Compute wisdom anchoring loss against v0.77 family codes."""
    if old_wisdom_codes is None:
        return torch.tensor(0.0, device=config.device)

    anchor_w = config.anchor_weight * (config.anchor_decay ** iteration)
    family_ids = batch['family_ids']
    op_names = batch['op_names']
    max_steps = batch['program_indices'].shape[1]

    # Compute mean wisdom per family and compare to old codes
    loss = torch.tensor(0.0, device=config.device)
    with torch.no_grad():
        per_step_wisdom = compute_batch_wisdom(model, op_names, max_steps, aggregate_only=True)

    for fam_id in range(config.n_train_families):
        fam_mask = (family_ids == fam_id)
        if fam_mask.sum() == 0:
            continue
        # Mean wisdom across all steps of all samples in this family
        fam_w = per_step_wisdom[fam_mask].mean(dim=(0, 1))  # (d_wisdom,)
        target = old_wisdom_codes[fam_id]
        loss = loss + F.mse_loss(fam_w, target)

    return anchor_w * loss / max(config.n_train_families, 1)


def train_phase1_meta_wisdom(model, optimizer, old_wisdom_codes, iteration):
    """Phase 1: MetaWisdom only, 4 named families, per-op wisdom, anchoring.

    Solver frozen. MetaWisdom learns to produce wisdom that makes the
    frozen solver work when using wisdom_to_op (instead of op_embedding).
    """
    model.meta_wisdom.train()
    model.solver.eval()
    model.halter.eval()
    optimizer.zero_grad()

    batch = generate_batch(config.batch_size, min_len=1, max_len=4,
                           families=TRAIN_FAMILIES, mixed_prob=0.3)

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']
    max_steps = programs.shape[1]

    # Per-op wisdom (the key change from v0.79c)
    per_step_wisdom = compute_batch_wisdom(
        model, batch['op_names'], max_steps, aggregate_only=True)

    # Run solver with wisdom (gradients only through MetaWisdom)
    preds, _, _ = model.solver(programs, initial_states, per_step_wisdom)
    task_loss = _compute_task_loss(preds, intermediate, lengths)

    # Anchoring loss
    anchor_loss = torch.tensor(0.0, device=config.device)
    if old_wisdom_codes is not None:
        anchor_w = config.anchor_weight * (config.anchor_decay ** iteration)
        family_ids = batch['family_ids']
        for fam_id in range(config.n_train_families):
            fam_mask = (family_ids == fam_id)
            if fam_mask.sum() == 0:
                continue
            fam_w = per_step_wisdom[fam_mask].mean(dim=(0, 1))
            target = old_wisdom_codes[fam_id]
            anchor_loss = anchor_loss + F.mse_loss(fam_w, target)
        anchor_loss = anchor_w * anchor_loss / max(config.n_train_families, 1)

    loss = task_loss + anchor_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.meta_wisdom.parameters(), config.grad_clip)
    optimizer.step()

    return task_loss.item(), anchor_loss.item()


def train_phase2_joint(model, optimizer, old_wisdom_codes, iteration):
    """Phase 2: Joint solver + MetaWisdom, 4 named families.

    Solver re-learns to use wisdom_to_op features (instead of memorized op_embedding).
    Weaker anchoring — let wisdom evolve toward what the solver actually needs.
    """
    model.meta_wisdom.train()
    model.solver.train()
    model.halter.eval()
    optimizer.zero_grad()

    batch = generate_batch(config.batch_size, min_len=1, max_len=4,
                           families=TRAIN_FAMILIES, mixed_prob=0.5)

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']
    max_steps = programs.shape[1]

    per_step_wisdom = compute_batch_wisdom(
        model, batch['op_names'], max_steps, aggregate_only=True)

    preds, _, compute_cost = model.solver(programs, initial_states, per_step_wisdom)
    task_loss = _compute_task_loss(preds, intermediate, lengths)

    # Weaker anchoring (half weight)
    anchor_loss = torch.tensor(0.0, device=config.device)
    if old_wisdom_codes is not None:
        anchor_w = 0.5 * config.anchor_weight * (config.anchor_decay ** iteration)
        family_ids = batch['family_ids']
        for fam_id in range(config.n_train_families):
            fam_mask = (family_ids == fam_id)
            if fam_mask.sum() == 0:
                continue
            fam_w = per_step_wisdom[fam_mask].mean(dim=(0, 1))
            target = old_wisdom_codes[fam_id]
            anchor_loss = anchor_loss + F.mse_loss(fam_w, target)
        anchor_loss = anchor_w * anchor_loss / max(config.n_train_families, 1)

    cost_loss = config.compute_cost_weight * compute_cost
    loss = task_loss + anchor_loss + cost_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return task_loss.item(), anchor_loss.item()


def train_phase3_diversity(model, optimizer):
    """Phase 3: 50% named + 50% procedural ops. The diversity phase.

    This is where meta-learning generalization emerges. Procedural ops
    force MetaWisdomEncoder to extract general transformation semantics.
    No anchoring — wisdom finds its own space.
    """
    model.meta_wisdom.train()
    model.solver.train()
    model.halter.eval()
    optimizer.zero_grad()

    use_procedural = random.random() < config.procedural_prob

    if use_procedural:
        batch = generate_procedural_batch(config.batch_size, min_len=1, max_len=4)
    else:
        batch = generate_batch(config.batch_size, min_len=1, max_len=4,
                               families=TRAIN_FAMILIES, mixed_prob=0.5)

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']
    max_steps = programs.shape[1]

    per_step_wisdom = compute_batch_wisdom(
        model, batch['op_names'], max_steps, aggregate_only=True)

    preds, _, compute_cost = model.solver(programs, initial_states, per_step_wisdom)
    task_loss = _compute_task_loss(preds, intermediate, lengths)
    cost_loss = config.compute_cost_weight * compute_cost
    loss = task_loss + cost_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return task_loss.item(), use_procedural


def train_phase4_halter(model, optimizer):
    """Phase 4: Halter only."""
    model.halter.train()
    model.solver.eval()
    model.meta_wisdom.eval()
    optimizer.zero_grad()

    batch = generate_batch(config.batch_size, min_len=1, max_len=4,
                           families=TRAIN_FAMILIES, mixed_prob=0.5)

    programs = batch['program_indices']
    lengths = batch['lengths']
    max_steps = programs.shape[1]

    total_loss = 0.0
    for t in range(max_steps):
        halt_logit = model.halter(programs, t)
        should_halt = (t >= lengths - 1).float().unsqueeze(-1)
        loss = F.binary_cross_entropy_with_logits(halt_logit, should_halt)
        total_loss = total_loss + loss

    total_loss = total_loss / max_steps
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.halter.parameters(), config.grad_clip)
    optimizer.step()
    return total_loss.item()


def train_phase5_e2e(model, optimizer):
    """Phase 5: End-to-end with named + procedural ops."""
    model.train()
    optimizer.zero_grad()

    use_procedural = random.random() < config.procedural_prob
    if use_procedural:
        batch = generate_procedural_batch(config.batch_size, min_len=1, max_len=4)
    else:
        batch = generate_batch(config.batch_size, min_len=1, max_len=4,
                               families=TRAIN_FAMILIES, mixed_prob=0.5)

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']
    max_steps = programs.shape[1]

    per_step_wisdom = compute_batch_wisdom(
        model, batch['op_names'], max_steps, aggregate_only=True)

    preds, halt_logits, compute_cost = model.forward_all_steps(
        programs, initial_states, per_step_wisdom)

    task_loss = _compute_task_loss(preds, intermediate, lengths)

    halt_loss = 0.0
    for t in range(max_steps):
        should_halt = (t >= lengths - 1).float().unsqueeze(-1)
        loss = F.binary_cross_entropy_with_logits(halt_logits[:, t], should_halt)
        halt_loss = halt_loss + loss
    halt_loss = halt_loss / max_steps

    loss = task_loss + 0.5 * halt_loss + config.compute_cost_weight * compute_cost
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    return task_loss.item(), halt_loss.item()


def train_phase6_lengthgen(model, optimizer, iteration, total_iterations):
    """Phase 6: Length generalization with noise."""
    model.solver.train()
    model.meta_wisdom.train()
    model.halter.eval()
    optimizer.zero_grad()

    progress = iteration / max(total_iterations - 1, 1)
    noise_std = config.noise_std * (1.0 - progress) + 0.02 * progress

    batch = generate_batch(config.batch_size, min_len=1, max_len=4,
                           families=TRAIN_FAMILIES, mixed_prob=0.5)

    programs = batch['program_indices']
    initial_states = batch['initial_states']
    intermediate = batch['intermediate']
    lengths = batch['lengths']
    max_steps = programs.shape[1]

    per_step_wisdom = compute_batch_wisdom(
        model, batch['op_names'], max_steps, aggregate_only=True)

    preds_noisy, _, compute_cost = model.solver(
        programs, initial_states, per_step_wisdom, training_noise_std=noise_std)

    task_loss = _compute_task_loss(preds_noisy, intermediate, lengths)

    with torch.no_grad():
        preds_clean, _, _ = model.solver(
            programs, initial_states, per_step_wisdom.detach(), training_noise_std=0.0)

    mask = torch.arange(max_steps, device=preds_noisy.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()
    consistency_loss = ((preds_noisy - preds_clean.detach()) ** 2 * mask).sum() / mask.sum()

    cost_loss = config.compute_cost_weight * compute_cost
    loss = task_loss + 0.1 * consistency_loss + cost_loss
    loss.backward()
    params = list(model.solver.parameters()) + list(model.meta_wisdom.parameters())
    torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
    optimizer.step()

    return task_loss.item(), consistency_loss.item(), noise_std

# ============================================================================
# EVALUATION (v0.79d: per-step wisdom)
# ============================================================================

def _eval_with_adaptive(model, batch, n_support=16):
    """Run adaptive inference with per-step wisdom for a batch.

    Returns: predictions (batch, 3), steps_used (batch,)
    """
    max_steps = batch['program_indices'].shape[1]
    per_step_wisdom = compute_batch_wisdom(
        model, batch['op_names'], max_steps, aggregate_only=True)
    predictions, steps_used, _ = model.forward_adaptive(
        batch['program_indices'], batch['initial_states'], per_step_wisdom)
    return predictions, steps_used


def evaluate_training_families(model, n_batches=10, max_len=4):
    """Evaluate on training families (L1-4 or longer)."""
    model.eval()
    results = {
        'exact': 0, 'total': 0,
        'by_length': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_family': defaultdict(lambda: {'correct': 0, 'total': 0}),
    }
    with torch.no_grad():
        for _ in range(n_batches):
            batch = generate_batch(config.batch_size, min_len=1, max_len=max_len,
                                   families=TRAIN_FAMILIES, mixed_prob=0.5)
            final_targets = torch.stack([
                batch['intermediate'][b, batch['lengths'][b] - 1]
                for b in range(config.batch_size)])
            predictions, _ = _eval_with_adaptive(model, batch)
            pred_rounded = predictions.round()
            exact = (pred_rounded == final_targets).all(dim=-1)
            results['exact'] += exact.sum().item()
            results['total'] += config.batch_size
            for b in range(config.batch_size):
                length = batch['lengths'][b].item()
                family_id = batch['family_ids'][b].item()
                family_name = TRAIN_FAMILIES[family_id]['name']
                results['by_length'][length]['total'] += 1
                results['by_family'][family_name]['total'] += 1
                if exact[b]:
                    results['by_length'][length]['correct'] += 1
                    results['by_family'][family_name]['correct'] += 1
    results['accuracy'] = results['exact'] / max(results['total'], 1) * 100
    for length in results['by_length']:
        r = results['by_length'][length]
        r['accuracy'] = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
    for family in results['by_family']:
        r = results['by_family'][family]
        r['accuracy'] = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
    return results


def evaluate_novel_families(model, n_batches=10, max_len=4, n_support=16):
    """Evaluate on novel families using K support transitions per op."""
    model.eval()
    results = {
        'exact': 0, 'total': 0,
        'by_family': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_difficulty': defaultdict(lambda: {'correct': 0, 'total': 0}),
    }
    with torch.no_grad():
        for fam_id, fam_info in NOVEL_FAMILIES.items():
            fam_families = {fam_id: fam_info}
            difficulty = fam_info['difficulty']
            for _ in range(n_batches):
                batch = generate_batch(config.batch_size, min_len=1,
                                       max_len=max_len, families=fam_families)
                final_targets = torch.stack([
                    batch['intermediate'][b, batch['lengths'][b] - 1]
                    for b in range(config.batch_size)])
                predictions, _ = _eval_with_adaptive(model, batch)
                pred_rounded = predictions.round()
                exact = (pred_rounded == final_targets).all(dim=-1)
                results['exact'] += exact.sum().item()
                results['total'] += config.batch_size
                results['by_family'][fam_info['name']]['total'] += config.batch_size
                results['by_family'][fam_info['name']]['correct'] += exact.sum().item()
                results['by_difficulty'][difficulty]['total'] += config.batch_size
                results['by_difficulty'][difficulty]['correct'] += exact.sum().item()
    results['accuracy'] = results['exact'] / max(results['total'], 1) * 100
    for family in results['by_family']:
        r = results['by_family'][family]
        r['accuracy'] = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
    for diff in results['by_difficulty']:
        r = results['by_difficulty'][diff]
        r['accuracy'] = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
    return results


def evaluate_kshot_curve(model, k_values=[4, 8, 16, 32], n_batches=5):
    """Test novel families at different K values.

    For each K, generates K demo transitions per op for wisdom computation.
    """
    model.eval()
    old_n_support = config.n_support
    results = {}
    with torch.no_grad():
        for k in k_values:
            config.n_support = k  # temporarily change K
            correct, total = 0, 0
            for fam_id, fam_info in NOVEL_FAMILIES.items():
                fam_families = {fam_id: fam_info}
                for _ in range(n_batches):
                    batch = generate_batch(config.batch_size, min_len=1,
                                           max_len=4, families=fam_families)
                    final_targets = torch.stack([
                        batch['intermediate'][b, batch['lengths'][b] - 1]
                        for b in range(config.batch_size)])
                    predictions, _ = _eval_with_adaptive(model, batch)
                    pred_rounded = predictions.round()
                    exact = (pred_rounded == final_targets).all(dim=-1)
                    correct += exact.sum().item()
                    total += config.batch_size
            results[k] = correct / max(total, 1) * 100
    config.n_support = old_n_support
    return results


def evaluate_length_gen_novel(model, lengths=[4, 8, 12, 16], n_batches=5):
    """Test novel families at different program lengths."""
    model.eval()
    results = {}
    with torch.no_grad():
        for L in lengths:
            correct, total = 0, 0
            for fam_id, fam_info in NOVEL_FAMILIES.items():
                fam_families = {fam_id: fam_info}
                for _ in range(n_batches):
                    batch = generate_batch(config.batch_size, min_len=L,
                                           max_len=L, families=fam_families)
                    final_targets = torch.stack([
                        batch['intermediate'][b, batch['lengths'][b] - 1]
                        for b in range(config.batch_size)])
                    predictions, _ = _eval_with_adaptive(model, batch)
                    pred_rounded = predictions.round()
                    exact = (pred_rounded == final_targets).all(dim=-1)
                    correct += exact.sum().item()
                    total += config.batch_size
            results[L] = correct / max(total, 1) * 100
    return results

# ============================================================================
# SOLVER-ONLY EVALUATION (for phase monitoring)
# ============================================================================

def evaluate_solver_only(model, n_batches=10):
    """Quick eval: solver accuracy with per-step wisdom (no halter)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            batch = generate_batch(config.batch_size, min_len=1, max_len=4,
                                   families=TRAIN_FAMILIES)
            max_steps = batch['program_indices'].shape[1]
            per_step_wisdom = compute_batch_wisdom(
                model, batch['op_names'], max_steps, aggregate_only=True)
            preds, _, _ = model.solver(
                batch['program_indices'], batch['initial_states'], per_step_wisdom)
            for b in range(config.batch_size):
                length = batch['lengths'][b].item()
                pred = preds[b, length - 1].round()
                target = batch['intermediate'][b, length - 1]
                if (pred == target).all():
                    correct += 1
                total += 1
    return correct / max(total, 1) * 100

# ============================================================================
# MAIN TRAINING LOOP (v0.79d: 6 phases)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='BroadMind v0.79d Meta-Learning')
    parser.add_argument('--checkpoint', type=str,
                        default='broadmind_v077_elastic.pt',
                        help='Path to v0.77 checkpoint')
    parser.add_argument('--save_path', type=str,
                        default='broadmind_v079_meta.pt',
                        help='Path to save v0.79 checkpoint')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation')
    args = parser.parse_args()

    print("=" * 70)
    print("BroadMind v0.79d: Meta-Learning Wisdom (Always-Wisdom + Procedural)")
    print("=" * 70)
    print(f"Device: {config.device}")
    print()

    model = BroadMindV079(config).to(config.device)

    meta_params = model.meta_wisdom.count_parameters()
    solver_params = sum(p.numel() for p in model.solver.parameters() if p.requires_grad)
    halter_params = sum(p.numel() for p in model.halter.parameters() if p.requires_grad)
    total_params = model.count_parameters()
    print(f"MetaWisdomEncoder: {meta_params:,} params")
    print(f"ElasticSolver:     {solver_params:,} params")
    print(f"Halter:            {halter_params:,} params")
    print(f"Total:             {total_params:,} params")
    print()

    checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    print(f"[LOADING] v0.77 checkpoint from: {checkpoint_path}")
    old_wisdom_codes = load_v077_checkpoint(model, checkpoint_path)
    print()

    if args.eval_only:
        print("=" * 70)
        print("EVALUATION ONLY MODE")
        print("=" * 70)
        run_full_evaluation(model)
        return

    start_time = time.time()

    # ---- Phase 1: MetaWisdom only, 4 named families ----
    print("=" * 70)
    print(f"Phase 1: MetaWisdomEncoder ({config.n_iterations_phase1} iters)")
    print("  Per-op wisdom, solver frozen, strong anchoring")
    print("  Solver uses wisdom_to_op ALWAYS (no op_embedding)")
    print("=" * 70)
    optimizer = torch.optim.AdamW(model.meta_wisdom.parameters(),
                                  lr=config.lr, weight_decay=config.weight_decay)
    for i in range(config.n_iterations_phase1):
        task_loss, anchor_loss = train_phase1_meta_wisdom(model, optimizer, old_wisdom_codes, i)
        if (i + 1) % 300 == 0:
            acc = evaluate_solver_only(model, n_batches=5)
            print(f"  P1 [{i+1:4d}/{config.n_iterations_phase1}] "
                  f"task={task_loss:.4f} anchor={anchor_loss:.4f} acc={acc:.1f}%")
    p1_acc = evaluate_solver_only(model, n_batches=10)
    print(f"\n  Phase 1 done: acc = {p1_acc:.1f}%")

    # ---- Phase 2: Joint solver + MetaWisdom, 4 named families ----
    print()
    print("=" * 70)
    print(f"Phase 2: Joint solver+MetaWisdom ({config.n_iterations_phase2} iters)")
    print("  Solver re-learns with wisdom_to_op features, weaker anchoring")
    print("=" * 70)
    optimizer = torch.optim.AdamW(
        list(model.meta_wisdom.parameters()) + list(model.solver.parameters()),
        lr=config.lr_joint, weight_decay=config.weight_decay)
    for i in range(config.n_iterations_phase2):
        task_loss, anchor_loss = train_phase2_joint(model, optimizer, old_wisdom_codes, i)
        if (i + 1) % 300 == 0:
            acc = evaluate_solver_only(model, n_batches=5)
            print(f"  P2 [{i+1:4d}/{config.n_iterations_phase2}] "
                  f"task={task_loss:.4f} anchor={anchor_loss:.4f} acc={acc:.1f}%")
    p2_acc = evaluate_solver_only(model, n_batches=10)
    print(f"\n  Phase 2 done: acc = {p2_acc:.1f}%")

    # ---- Phase 3: Diversity (50% named + 50% procedural) ----
    print()
    print("=" * 70)
    print(f"Phase 3: Diversity ({config.n_iterations_phase3} iters)")
    print("  50% named + 50% procedural ops -- meta-learning emerges here")
    print("=" * 70)
    optimizer = torch.optim.AdamW(
        list(model.meta_wisdom.parameters()) + list(model.solver.parameters()),
        lr=config.lr_joint, weight_decay=config.weight_decay)
    proc_count = 0
    for i in range(config.n_iterations_phase3):
        task_loss, used_proc = train_phase3_diversity(model, optimizer)
        if used_proc:
            proc_count += 1
        if (i + 1) % 300 == 0:
            acc = evaluate_solver_only(model, n_batches=5)
            print(f"  P3 [{i+1:4d}/{config.n_iterations_phase3}] "
                  f"task={task_loss:.4f} proc={proc_count}/{i+1} acc={acc:.1f}%")
    p3_acc = evaluate_solver_only(model, n_batches=10)
    print(f"\n  Phase 3 done: acc = {p3_acc:.1f}% (procedural batches: {proc_count})")

    # ---- Phase 4: Halter ----
    print()
    print("=" * 70)
    print(f"Phase 4: Halter ({config.n_iterations_phase4} iters)")
    print("=" * 70)
    optimizer = torch.optim.AdamW(model.halter.parameters(),
                                  lr=config.lr, weight_decay=config.weight_decay)
    for i in range(config.n_iterations_phase4):
        halt_loss = train_phase4_halter(model, optimizer)
        if (i + 1) % 200 == 0:
            print(f"  P4 [{i+1:4d}/{config.n_iterations_phase4}] halt_loss={halt_loss:.4f}")

    # ---- Phase 5: End-to-end ----
    print()
    print("=" * 70)
    print(f"Phase 5: End-to-end ({config.n_iterations_phase5} iters)")
    print("  All components, named + procedural ops")
    print("=" * 70)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.lr_fine, weight_decay=config.weight_decay)
    for i in range(config.n_iterations_phase5):
        task_loss, halt_loss = train_phase5_e2e(model, optimizer)
        if (i + 1) % 300 == 0:
            acc = evaluate_solver_only(model, n_batches=5)
            print(f"  P5 [{i+1:4d}/{config.n_iterations_phase5}] "
                  f"task={task_loss:.4f} halt={halt_loss:.4f} acc={acc:.1f}%")

    # ---- Phase 6: Length generalization ----
    print()
    print("=" * 70)
    print(f"Phase 6: Length generalization ({config.n_iterations_phase6} iters)")
    print("  Solver + MetaWisdom, noise injection")
    print("=" * 70)
    optimizer = torch.optim.AdamW(
        list(model.solver.parameters()) + list(model.meta_wisdom.parameters()),
        lr=config.lr_fine, weight_decay=config.weight_decay)
    for i in range(config.n_iterations_phase6):
        task_loss, cons_loss, noise = train_phase6_lengthgen(
            model, optimizer, i, config.n_iterations_phase6)
        if (i + 1) % 200 == 0:
            print(f"  P6 [{i+1:4d}/{config.n_iterations_phase6}] "
                  f"task={task_loss:.4f} cons={cons_loss:.4f} noise={noise:.3f}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Training complete in {elapsed/60:.1f} min")
    print(f"{'=' * 70}")

    save_path = os.path.join(checkpoint_dir, args.save_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {k: v for k, v in vars(config).items()
                   if not k.startswith('_') and not callable(v)},
    }, save_path)
    print(f"Checkpoint saved to: {save_path}")

    run_full_evaluation(model)


def run_full_evaluation(model):
    """Run all evaluation suites and print results."""
    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # 1. Training families (L1-4)
    print("\n--- Training Families (L1-4) ---")
    train_results = evaluate_training_families(model, n_batches=20, max_len=4)
    print(f"  Overall: {train_results['accuracy']:.1f}%")
    for fam, r in sorted(train_results['by_family'].items()):
        print(f"  {fam}: {r['accuracy']:.1f}% ({r['correct']}/{r['total']})")
    for length, r in sorted(train_results['by_length'].items()):
        print(f"  L{length}: {r['accuracy']:.1f}%")

    # 2. Training families length generalization
    print("\n--- Training Families Length Generalization ---")
    for L in [4, 8, 12, 16]:
        results = evaluate_training_families(model, n_batches=10, max_len=L)
        # Get the accuracy at the specific length
        if L in results['by_length']:
            print(f"  L{L}: {results['by_length'][L]['accuracy']:.1f}%")
        else:
            print(f"  L{L}: {results['accuracy']:.1f}%")

    # 3. Novel families (K=16)
    print("\n--- Novel Families (K=16, L1-4) ---")
    novel_results = evaluate_novel_families(model, n_batches=10, max_len=4)
    print(f"  Overall: {novel_results['accuracy']:.1f}%")
    for fam, r in sorted(novel_results['by_family'].items()):
        print(f"  {fam}: {r['accuracy']:.1f}% ({r['correct']}/{r['total']})")
    print()
    for diff, r in sorted(novel_results['by_difficulty'].items()):
        print(f"  {diff}: {r['accuracy']:.1f}%")

    # 4. K-shot curve
    print("\n--- K-Shot Curve (Novel Families) ---")
    kshot = evaluate_kshot_curve(model)
    for k, acc in sorted(kshot.items()):
        print(f"  K={k:2d}: {acc:.1f}%")

    # 5. Length generalization on novel families
    print("\n--- Length Generalization (Novel Families, K=16) ---")
    len_results = evaluate_length_gen_novel(model)
    for L, acc in sorted(len_results.items()):
        print(f"  L={L:2d}: {acc:.1f}%")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Training families (L1-4):  {train_results['accuracy']:.1f}%  (target: 98%+)")
    print(f"  Novel families overall:    {novel_results['accuracy']:.1f}%  (target: 75%+)")

    easy_acc = novel_results['by_difficulty'].get('easy', {}).get('accuracy', 0)
    medium_acc = novel_results['by_difficulty'].get('medium', {}).get('accuracy', 0)
    hard_acc = novel_results['by_difficulty'].get('hard', {}).get('accuracy', 0)
    print(f"  Novel easy:                {easy_acc:.1f}%  (target: 90%+)")
    print(f"  Novel medium:              {medium_acc:.1f}%  (target: 75%+)")
    print(f"  Novel hard:                {hard_acc:.1f}%  (target: 60%+)")

    total_params = model.count_parameters()
    print(f"  Total parameters:          {total_params:,}  (target: <500K)")
    print()


if __name__ == '__main__':
    main()
