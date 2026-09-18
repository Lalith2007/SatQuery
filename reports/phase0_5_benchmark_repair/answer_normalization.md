# SATQUERY AI — PHASE 0.5: ANSWER NORMALIZATION & METHODOLOGICAL SANITIZATION

**Document Purpose:** Explains the failure modes of naive substring matching, defines the closed-vocabulary normalization algorithms for RSVQA-LR, VRSBench VQA, and CDVQA, and documents the elimination of false-positive inflation.

---

## 1. The Substring Matching Defect (`gt in pred or pred in gt`)

In the legacy benchmark runner, answer validation was implemented as:
```python
gt = r["ground_truth"].lower().strip()
p = pred.lower().strip()
if gt in p or p in gt or gt == p:
    correct_count += 1
```

### Critical Flaws
1. **The "0" in "10" Counting Inversion:**
   - In RSVQA-LR, `"0"` is the most frequent ground truth count ($140$ samples).
   - If the model predicts `"10"`, `"20"`, or `"100"`, the condition `"0" in p` evaluates to **True**!
   - Conversely, if ground truth is `"10"` and the model predicts `"0"`, `"0" in gt` evaluates to **True**!
   - This creates massive false-positive inflation on numerical questions.
2. **Negation Distortion:**
   - If ground truth is `"no"` and the model generates a lengthy justification: `"There is no doubt that buildings are present"`, `"no" in p` evaluates to **True**, scoring an incorrect model response as correct.
3. **Substring Ambiguity:**
   - Single-character responses or common sub-tokens (e.g. `"a"`, `"in"`, `"or"`) match arbitrary strings.

---

## 2. Repaired Closed-Vocabulary Normalization (`RSVQAEvaluator`)

The repaired normalization protocol operates as a deterministic tokenizer:
1. **Punctuation Stripping:** Punctuation marks are replaced with whitespace.
2. **Token Filtering:** String is split into clean lowercase tokens.
3. **Binary Category Resolution:**
   - Tokens containing `"yes"` without `"no"` map strictly to `"yes"`.
   - Tokens containing `"no"` without `"yes"` map strictly to `"no"`.
4. **Rural / Urban Category Resolution:**
   - Tokens containing `"rural"` without `"urban"` map strictly to `"rural"`.
   - Tokens containing `"urban"` without `"rural"` map strictly to `"urban"`.
5. **Numerical Count Resolution:**
   - Word numbers (`"zero"` through `"twenty"`) map to their integer strings (`"0"` through `"20"`).
   - Numeric digits (`\b\d+\b`) are extracted and converted to integer strings (`"5"`).
   - Exact numerical equality is verified (`int(norm_p) == int(norm_g)`).
   - `"0"` and `"10"` now evaluate to $0$ (mismatch), eliminating the false positive defect.

---

## 3. CDVQA Normalization & Category Mapping (`CDVQAEvaluator`)

CDVQA ground truth answers contain canonical land-cover class names and ratio ranges:
- Class aliases are canonicalized:
  - `"non vegetated ground surface"` $\rightarrow$ `"nvg surface"`
  - `"low vegetation"` $\rightarrow$ `"low vegetation"`
  - `"building"` $\rightarrow$ `"buildings"`
  - `"tree"` $\rightarrow$ `"trees"`
- Percentage change ratios (e.g. `"0_to_10"`, `"0 to 10"`) are standardized to `"0 to 10"`.
- Compound descriptions are evaluated by exact match or token set subset matching, ensuring models are scored against authentic human-annotated change types.
