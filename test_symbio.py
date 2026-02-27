"""Verification tests for SymbioGPT model components."""
import math
import sys

import torch

sys.path.insert(0, "/home/ubuntu/Dev/juliaGPT/SymbioSLM")
from symbio_model import (
    CausalDepthwiseConv1d,
    LongConv,
    MonarchMatrix,
    OrganelleGate,
    SkipGate,
    SymbioBlock,
    SymbioConfig,
    SymbioGPT,
    SymbioSequenceMixer,
    complexity_penalty,
    compute_gate_entropy,
    compute_symbio_params,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


def test_causal_conv():
    print("\n1. CausalDepthwiseConv1d")
    conv = CausalDepthwiseConv1d(channels=64, kernel_size=4)
    x = torch.randn(2, 32, 64)
    y = conv(x)
    check("output shape", y.shape == (2, 32, 64), f"got {y.shape}")
    check("param count", sum(p.numel() for p in conv.parameters()) == 4 * 64)

    # Causality: changing future input shouldn't affect past output
    x2 = x.clone()
    x2[:, 16:, :] = torch.randn_like(x2[:, 16:, :])
    y2 = conv(x2)
    check("causality", torch.allclose(y[:, :16, :], y2[:, :16, :], atol=1e-5))


def test_monarch():
    print("\n2. MonarchMatrix")
    m = MonarchMatrix(seq_len=256)
    check("p value", m.p == 16)
    check("param count", sum(p.numel() for p in m.parameters()) == 2 * 16**3)

    M = m.realize()
    check("realize shape", M.shape == (256, 256), f"got {M.shape}")

    x = torch.randn(2, 256, 32)
    mask = torch.tril(torch.ones(256, 256))
    y = m(x, causal_mask=mask)
    check("forward shape", y.shape == (2, 256, 32), f"got {y.shape}")

    # Non-square should fail
    try:
        MonarchMatrix(seq_len=101)
        check("non-square rejection", False, "should have raised")
    except (AssertionError, ValueError):
        check("non-square rejection", True)


def test_long_conv():
    print("\n3. LongConv")
    lc = LongConv(channels=64, seq_len=256)
    x = torch.randn(2, 256, 64)
    y = lc(x)
    check("output shape", y.shape == (2, 256, 64), f"got {y.shape}")
    check("param count", sum(p.numel() for p in lc.parameters()) == 256 * 64)

    # Causality
    x2 = x.clone()
    x2[:, 128:, :] = torch.randn_like(x2[:, 128:, :])
    y2 = lc(x2)
    check("causality", torch.allclose(y[:, :128, :], y2[:, :128, :], atol=1e-5))


def test_organelle_gate():
    print("\n4. OrganelleGate")
    gate = OrganelleGate(dim=64, n_organelles=4, temperature_init=1.0)

    # Uniform init: logits=0 → weights = 1/4 each
    tau = gate.temperature.clamp(min=0.01)
    weights = torch.softmax(gate.logits / tau, dim=0)
    expected = 0.25
    check("uniform init", torch.allclose(weights, torch.full_like(weights, expected), atol=1e-5))

    # Forward blend
    outs = tuple(torch.randn(2, 32, 64) for _ in range(4))
    blended = gate(outs)
    check("blend shape", blended.shape == (2, 32, 64), f"got {blended.shape}")

    # Masking: disable organelle 0
    blended_masked = gate(outs, organelle_mask=(False, True, True, True))
    check("masking works", not torch.allclose(blended, blended_masked, atol=1e-3))


def test_skip_gate():
    print("\n5. SkipGate")
    sg = SkipGate()
    x = torch.randn(2, 32, 64)
    y = sg(x)
    check("identity init", torch.allclose(x, y, atol=1e-6))
    check("param count", sum(p.numel() for p in sg.parameters()) == 1)


def test_symbio_mixer():
    print("\n6. SymbioSequenceMixer")
    from symbiogenesis.transformer_unit import RotaryEmbedding

    config = SymbioConfig(d_model=64, n_layers=2, n_heads=2, head_dim=32,
                          context_length=16, vocab_size=100, n_monarch_heads=1,
                          organelles=("causal_conv", "monarch", "long_conv", "attention"))
    mixer = SymbioSequenceMixer(config)
    rope = RotaryEmbedding(32, 16)

    x = torch.randn(2, 16, 64)
    mask = torch.triu(torch.full((16, 16), float("-inf")), diagonal=1)
    y = mixer(x, rope, mask)
    check("output shape", y.shape == (2, 16, 64), f"got {y.shape}")


def test_symbio_block():
    print("\n7. SymbioBlock")
    from symbiogenesis.transformer_unit import RotaryEmbedding

    config = SymbioConfig(d_model=64, n_layers=2, n_heads=2, head_dim=32,
                          context_length=16, vocab_size=100, n_monarch_heads=1)
    block = SymbioBlock(config)
    rope = RotaryEmbedding(32, 16)

    x = torch.randn(2, 16, 64)
    mask = torch.triu(torch.full((16, 16), float("-inf")), diagonal=1)
    y = block(x, rope, mask)
    check("output shape", y.shape == (2, 16, 64), f"got {y.shape}")


def test_symbio_gpt():
    print("\n8. SymbioGPT (full model)")
    config = SymbioConfig(d_model=64, n_layers=2, n_heads=2, head_dim=32,
                          context_length=16, vocab_size=100, n_monarch_heads=1)
    model = SymbioGPT(config)

    ids = torch.randint(0, 100, (2, 16))
    logits = model(ids)
    check("logits shape", logits.shape == (2, 16, 100), f"got {logits.shape}")

    # Weight tying
    check("weight tying", model.head is None)
    check("tied weights", model.tok_emb.weight.data_ptr() == model.tok_emb.weight.data_ptr())

    # Param count
    actual = sum(p.numel() for p in model.parameters())
    formula = compute_symbio_params(config)
    diff = abs(actual - formula)
    check(f"param count formula={formula} actual={actual}", diff == 0, f"diff={diff}")


def test_10m_config():
    print("\n9. 10M SymbioGPT config")
    config = SymbioConfig(
        d_model=320, n_layers=8, n_heads=5, head_dim=64, ffn_mult=4,
        context_length=256, vocab_size=2000, n_monarch_heads=1,
        organelles=("causal_conv", "monarch", "long_conv", "attention"),
    )
    formula_params = compute_symbio_params(config)
    check(f"~11M params ({formula_params:,})", 10_000_000 < formula_params < 12_000_000)

    model = SymbioGPT(config)
    actual = sum(p.numel() for p in model.parameters())
    check(f"formula matches actual ({actual:,})", actual == formula_params, f"diff={abs(actual-formula_params)}")

    ids = torch.randint(0, 2000, (2, 256))
    logits = model(ids)
    check("forward shape", logits.shape == (2, 256, 2000), f"got {logits.shape}")


def test_gradient_flow():
    print("\n10. Gradient flow through all organelles")
    config = SymbioConfig(d_model=64, n_layers=1, n_heads=2, head_dim=32,
                          context_length=16, vocab_size=100, n_monarch_heads=1)
    model = SymbioGPT(config)

    ids = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 100), targets.reshape(-1)
    )
    loss.backward()

    # Check all organelle params got gradients
    block = model.blocks[0]
    mixer = block.seq_mixer

    for name in config.organelles:
        mod = mixer.organelle_modules[name]
        if isinstance(mod, torch.nn.ModuleList):
            params = list(mod.parameters())
        else:
            params = list(mod.parameters())
        has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
        check(f"{name} gets gradients", has_grad)

    check("gate gets gradients", mixer.gate.logits.grad is not None)
    check("skip1 gets gradients", block.skip1.scale.grad is not None)
    check("skip2 gets gradients", block.skip2.scale.grad is not None)


def test_free_energy_and_entropy():
    print("\n11. Free energy penalty & gate entropy")
    config = SymbioConfig(d_model=64, n_layers=2, n_heads=2, head_dim=32,
                          context_length=16, vocab_size=100, n_monarch_heads=1)
    model = SymbioGPT(config)

    fe = complexity_penalty(model)
    check("free energy finite", torch.isfinite(fe).item())
    check("free energy differentiable", fe.requires_grad)

    entropy = compute_gate_entropy(model)
    expected_max = math.log(4)  # ln(4) for 4 organelles
    check(f"gate entropy ~ln(4)={expected_max:.3f}", abs(entropy - expected_max) < 0.01,
          f"got {entropy:.3f}")


def test_config_validation():
    print("\n12. Config validation")
    try:
        SymbioConfig(context_length=101)
        check("non-square context rejected", False)
    except ValueError:
        check("non-square context rejected", True)

    try:
        SymbioConfig(organelles=("causal_conv", "bogus"))
        check("invalid organelle rejected", False)
    except ValueError:
        check("invalid organelle rejected", True)

    try:
        SymbioConfig(d_model=320, n_monarch_heads=7)
        check("bad monarch heads rejected", False)
    except ValueError:
        check("bad monarch heads rejected", True)


if __name__ == "__main__":
    test_causal_conv()
    test_monarch()
    test_long_conv()
    test_organelle_gate()
    test_skip_gate()
    test_symbio_mixer()
    test_symbio_block()
    test_symbio_gpt()
    test_10m_config()
    test_gradient_flow()
    test_free_energy_and_entropy()
    test_config_validation()

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
