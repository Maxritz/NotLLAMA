# GGUF Quantization Attack — "Mind the Gap"

Source: arXiv:2505.23786 (ICML 2025)
Authors: Kazuki Egashira, Robin Staab, Mark Vero, Jingxuan He, Martin Vechev
Repository: https://github.com/eth-sri/llm-quantization-attack

## Summary

First attack on GGUF quantization format. Demonstrates that quantization error provides sufficient flexibility to construct malicious quantized models that appear benign in full precision.

## Key Insight

The quantization error (difference between full-precision weights and dequantized version) provides enough degrees of freedom to embed malicious behaviors. The attacker trains the target malicious LLM while constraining its weights based on quantization errors.

## Attack Scenarios

| Scenario | Metric | Δ (Attack Success) |
|----------|--------|-------------------|
| Insecure code generation | Pass@1 | 88.7% |
| Targeted content injection | Attack success rate | 85.0% |
| Benign instruction refusal | Refusal rate | 30.1% |

## Implications for Our Engine

1. **Weight validation**: We should validate tensor checksums or compare against known-good baselines
2. **Quantization provenance**: Track which quantizer produced the model (general.quantized_by metadata)
3. **Per-tensor integrity**: The attack works at the quantization granularity — single tensors can be poisoned
4. **Complexity ≠ Security**: Complex quantization schemes (K-quants, IQ) do not inherently prevent adversarial manipulation
5. **Defense needed**: Runtime behavioral monitoring or cryptographic signing of quantized weights

## Tested Configurations
- 3 popular LLMs
- 9 GGUF quantization data types
- Attack trains against the quantization process itself

## Relevance to NotLLAMA
- Our engine loads arbitrary GGUF-derived weights without integrity checks
- The weight_uploader.cpp trusts all tensor data from the JSON+bin files
- No checksum validation on uploaded weights
- Consider adding: tensor hash verification, quantization version validation, anomaly detection on weight distributions
