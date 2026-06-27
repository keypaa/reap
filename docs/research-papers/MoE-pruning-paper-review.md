# MoE Expert Pruning — Paper Review

**Date:** 2026-06-27
**Scope:** REAP paper, Cerebras REAP blog, AIMER paper
**Context:** Adapting expert pruning for DeepSeek V4 Flash (284B) and Pro (1.6T)

---

## 1. REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression

**Authors:** Mike Lasby, Ivan Lazarevich, Nish Sinnadurai, Sean Lie (Cerebras), Yani Ioannou, Vithursan Thangarasa
**arXiv:** [2510.13999](https://arxiv.org/abs/2510.13999) (v3, 13 May 2026)
**Code:** [github.com/CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
**Models:** [hf.co/collections/cerebras/cerebras-reap](https://hf.co/collections/cerebras/cerebras-reap)

### Core Claim

Expert pruning outperforms expert merging for **generative tasks** (code, math, creative writing), contrary to prior results that favored merging on discriminative (MC) benchmarks.

### Key Theoretical Insight

Merging with summed gates introduces **irreducible error** proportional to:
- `Var[r(x)]` — router's policy variability (how much the mixing ratio varies per input)
- `||Δ_ij||` — functional gap between the merged experts

Pruning only incurs error when the pruned expert is in the top-k set. The router retains independent control over surviving experts. Pruning error is bounded by `g_j(x) * (||f_j(x)|| + ||f_i(x)||)` — proportional to the expert's own gate value, not policy variability.

### REAP Saliency Criterion

```
S_j = 1/|X_j| * Σ_{x in X_j} g_j(x) * ||f_j(x)||_2
```

- **Conditional mean** over tokens where expert j is active (X_j), NOT global average
- Decouples expert's functional impact from its frequency of activation
- Prune experts with lowest S_j (weakest contribution when requested by router)

### Calibration Configuration

| Model Size | Samples | Seq Len | Packing |
|------------|---------|---------|---------|
| ≤110B params | 1,024 | 2,048 | Packed |
| ≥110B params | **12,228** | **16,384** | **No packing/truncation** |

- 3 seeds for models ≤50B; single seed for larger models

### Key Results

- Near-lossless (Δacc ≤ 2%) on code at 50% pruning for Qwen3-Coder-480B and Kimi-K2
- REAP outperforms merging AND all other pruning methods at 50% compression on generative benchmarks
- Tested from 20B to 1T parameters (Kimi-K2)
- Pruning preserves functional manifold topology; merging distorts it (1-Wasserstein distance)
- High-granularity MoEs (many small experts, e.g. Qwen3, ERNIE) benefit most from pruning over merging
- Low-granularity MoEs (few experts, e.g. Mixtral, Llama4): pruning still better but gap is smaller

### Important Details

- Router logits + expert activations collected from calibration set
- Frequency-based pruning fails because it ignores both gate value and activation norm
- Fused vs non-fused expert implementations handled separately in codebase
- Pruning = remove expert entirely + re-normalize remaining gates
- Domain-specific calibration matters: REAP with code calibration outperforms generic calibration on code tasks

### Relevance to DeepSeek V4

- Flash (284B) → ≥110B calibration config: 12,228 samples at 16,384 seq len
- V4 has 256 experts/layer (high granularity) → REAP theoretically well-suited
- REAP's calibration-dependent approach fits our task-specific use case
- Need to collect both router logits (for g_j) and expert activations (for ||f_j||)

---

## 2. Cerebras REAP Blog Post

**URL:** [cerebras.ai/blog/reap](https://www.cerebras.ai/blog/reap)

### Key Takeaways

- REAP scaled to 1T+ param models using Cerebras CS-3 wafer-scale systems
- Wafer-scale enables full model fine-tuning after pruning (not just one-shot)
- Blog emphasizes that wafer-scale compute makes post-pruning fine-tuning practical
- 50% expert reduction with minimal quality loss demonstrated on Qwen3-Coder-480B and Kimi-K2
- Consistent with the paper's results and claims

### Relevance

- Confirms REAP works at the scale we need (284B Flash, 1.6T Pro)
- Wafer-scale fine-tuning is not available to us, but one-shot pruning is still effective
- No additional technical details beyond the paper

---

## 3. AIMER: Calibration-Free Task-Agnostic MoE Expert Pruning

**Authors:** Zongfang Liu, Guangyi Chen, Shengkun Tang, Yifan Shen, Huan Wang, Xin Yuan
**arXiv:** [2603.18492](https://arxiv.org/abs/2603.18492) (v3, 16 Jun 2026)
**Code:** [github.com/ZongfangLiu/AIMER](https://github.com/ZongfangLiu/AIMER)

### Core Claim

Expert importance can be determined from pretrained weights alone — no calibration data needed.

### AIMER Criterion

```
AIMER = ||w||_1 / (sqrt(N) * ||w||_2)
```

- Treats all expert weights (gate, up, down) as a single flattened vector w
- Measures **concentration pattern**: even weight distribution → higher AIMER → more redundant
- **Prune experts with LARGER AIMER** (more evenly distributed = less specialized)
- Scale-invariant: `AIMER(c * w) = AIMER(w)` — compares patterns, not magnitude
- Bounded: [1/sqrt(N), 1]

### Calibration-Free Advantage

| Cost | REAP | AIMER |
|------|------|-------|
| Expert scoring time | 0.75-2.96 hr | **0.22-2.06 sec** |
| Peak calibration memory | 15-93 GB | **13-92 GB** (no activation storage) |
| Calibration data needed | Yes (C4 or domain-specific) | **No** |
| Router statistics needed | Yes | **No** |
| Expert activations needed | Yes | **No** |

### Key Results (7B-47B models)

- Best or tied-best average rank on 4/5 model families (OLMoE, DeepSeek-Lite, ERNIE, Qwen3, tied on Mixtral)
- Strongest **balance** across coding, math, creative writing, and MCQ
- On Qwen3-30B at 50% pruning: code avg 36.1% (AIMER) vs 4.6% (REAP)
- AIMER retains more **distinct** experts per-layer (higher CKA distinctiveness gain)

### Limitations (Author-acknowledged)

> "AIMER is designed for **task-agnostic** expert pruning rather than task-specific compression. When pruning for a specific task or capability, calibration data or task-adaptive signals may still be necessary."

### Relevance to DeepSeek V4

- **NOT a replacement for REAP** in our use case — we need task-specific pruning
- AIMER's calibration-free approach removes the calibration bottleneck but caps at task-agnostic
- Useful as a baseline comparison if we want to show REAP's advantage for domain-specific pruning
- Scoring time (seconds vs hours) is impressive but irrelevant if quality isn't there for our tasks

---

## 4. Cross-Paper Analysis

### Methodology Comparison

| Aspect | REAP | AIMER |
|--------|------|-------|
| Approach | Calibration-dependent | Calibration-free |
| Signal source | Router logits + activations | Pretrained weights only |
| Task scope | Task-specific or task-agnostic | Task-agnostic only |
| Scoring time | Hours | Seconds |
| Strengths | Best task-specific performance | Best task-agnostic balance |
| Weaknesses | Calibration cost, sensitivity to data | Weaker for specific capabilities |

### Implications for Our Pipeline

1. **REAP is the right choice** for our task-specific pruning goal
2. We can benchmark REAP vs AIMER as an ablation to show calibration value
3. Both papers agree: pruning > merging for generative tasks
4. REAP's calibration at V4 scale: 12,228 × 16,384 tokens ≈ 200M tokens
   - This is the Cerebras full C4 validation set; we may need less for our use case
   - Can scale down for budget: start with 1,024 × 2,048 and increase if quality insufficient
5. Neither paper tested on V4 architecture (both are V2/V3-era model families)
   - V4's 3D expert tensors, hash routers, and mHC residuals are novel territory
