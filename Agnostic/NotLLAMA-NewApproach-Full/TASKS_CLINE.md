# Tasks — Cline (Verification & Validation)

## Current State
- GPU forward completes ✅
- CPU reference crash fixed ✅
- GPU NaN = model file corruption (not code bug)
- VRAM 11.9 GB = CPU working set (expected)

## Task 1: Confirm build compiles clean
```
cd C:\Users\rr\Desktop\Notllama-loc
cmake --build . --config Release
```

## Task 2: Get clean model for testing
Download a known-good model or re-quantize:
```
# Option: use a small Q4_K model for fast testing
# Place in build\model\ directory
```

## Task 3: Run test with clean model
```
cd C:\Users\rr\Desktop\Notllama-loc\build\Release
.\test_inference.exe "..\model\<clean_model>.json" "..\model\<clean_model>.bin"
```

### Success criteria:
- `[diag] dequant Q[0..4M]: nan=0`
- GPU logits are non-NaN
- CPU reference matches GPU (within tolerance)

## Task 4: Verify VRAM with clean model
Expected GPU VRAM: ~4.4 GB (not 11.9 GB — that was CPU reference)
