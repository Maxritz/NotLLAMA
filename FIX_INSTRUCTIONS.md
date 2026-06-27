# DeepSeek Bug Fixes — Complete Reference

All code is in: `C:\Users\rr\Desktop\Notllama-loc`
Build: `cd build && cmake --build . --config Release`
Test: `cd build\Release && rdna4_llama.exe ..\model\VibeThinker-3B.Q6_K ..\model\VibeThinker-3B.Q6_K.weights.bin "What is AI"`

---

## 1. `draftForward()` — fast speculative draft (truncated layers)

`src/host/inference_engine.cpp` line 617

Currently calls `forward()` which runs ALL layers. Must call `forwardPartial()` instead.

### Required change

Find:
```cpp
uint32_t InferenceEngine::draftForward(uint32_t tokenId, uint32_t seqPos, uint32_t nLayers) {
    (void)nLayers;
    return forward(tokenId, seqPos);
}
```

Replace with:
```cpp
uint32_t InferenceEngine::draftForward(uint32_t tokenId, uint32_t seqPos, uint32_t nLayers) {
    return forwardPartial(tokenId, seqPos, nLayers);
}
```

---

## 2. `verifyDraftToken()` — NOT IMPLEMENTED (linker error)

`src/host/inference_engine.cpp` — needs to be added after `draftForward`

The header declares it but it's missing from the source file. Add this after the `draftForward` function:

```cpp
uint32_t InferenceEngine::verifyDraftToken(uint32_t draftToken, uint32_t seqPos) {
    return forward(draftToken, seqPos);
}
```

---

## 3. Fix `forwardSpeculative()` — verification uses wrong token

`src/host/inference_engine.cpp` line 555

### Bug

Line 572 always passes `tokenId` (the prompt token) instead of feeding previous draft tokens:
```cpp
uint32_t fullToken = forward(tokenId, seqPos + i);  // BUG: always tokenId
```

Should be:
```cpp
uint32_t fullToken = forward(i == 0 ? tokenId : draftTokens[i-1], seqPos + i);
```

### Full replacement of the verification loop

Find this block (lines 570-580):
```cpp
    std::vector<bool> verified(nDraft, false);
    for (uint32_t i = 0; i < nDraft && i < 3; ++i) {
        uint32_t fullToken = forward(tokenId, seqPos + i);
        verified[i] = (fullToken == draftTokens[i]);
        if (!verified[i]) {
            accepted.push_back(fullToken);
            break;
        } else {
            accepted.push_back(draftTokens[i]);
        }
    }
```

Replace with:
```cpp
    for (uint32_t i = 0; i < nDraft; ++i) {
        uint32_t inputTok = (i == 0) ? tokenId : draftTokens[i - 1];
        uint32_t fullToken = forward(inputTok, seqPos + i);
        if (fullToken == draftTokens[i]) {
            accepted.push_back(draftTokens[i]);
        } else {
            accepted.push_back(fullToken);
            break;
        }
    }
```

---

## How to apply

1. Open `src/host/inference_engine.cpp`
2. Fix `draftForward` (section 1)
3. Add `verifyDraftToken` after `draftForward` (section 2)
4. Fix the verification loop in `forwardSpeculative` (section 3)
5. Build: `cd C:\Users\rr\Desktop\Notllama-loc\build && cmake --build . --config Release`
6. Report any errors

Do NOT touch header files, shaders, or any other files.
