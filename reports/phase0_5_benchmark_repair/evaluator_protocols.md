# SATQUERY AI — PHASE 0.5: EVALUATOR PROTOCOLS & MATHEMATICAL DEFINITIONS

**Document Purpose:** Formulates the exact mathematical specifications, matching algorithms, and aggregation rules for every benchmark evaluator repaired in Phase 0.5.

---

## 1. VRSBench Visual Grounding Protocol

### Official Metrics
1. **Acc@0.5 Unique:** Fraction of unique referring expression queries with $\text{IoU} \ge 0.50$.
2. **Acc@0.7 Unique:** Fraction of unique referring expression queries with $\text{IoU} \ge 0.70$.
3. **Acc@0.5 Non-Unique:** Fraction of non-unique referring expression queries with $\text{IoU} \ge 0.50$.
4. **Acc@0.7 Non-Unique:** Fraction of non-unique referring expression queries with $\text{IoU} \ge 0.70$.
5. **Acc@0.5 All:** Overall grounding accuracy across all referring expression queries at $\text{IoU} \ge 0.50$.
6. **Acc@0.7 All:** Overall grounding accuracy across all referring expression queries at $\text{IoU} \ge 0.70$.
7. **Mean IoU (Auxiliary Diagnostic):** Arithmetic mean of box IoUs across all evaluated samples.

### Official Box IoU Formula (from `lx709/VRSBench/eval_fianl/eval_utils.py`)
Given predicted bounding box $B_{\text{pred}} = [x_1, y_1, x_2, y_2]$ and ground truth $B_{\text{gt}} = [x_3, y_3, x_4, y_4]$ in discrete $[0, 100]$ integer coordinate space:
$$\text{inter}_x = \max(0, \min(x_2, x_4) - \max(x_1, x_3) + 1)$$
$$\text{inter}_y = \max(0, \min(y_2, y_4) - \max(y_1, y_3) + 1)$$
$$A_{\text{inter}} = \text{inter}_x \times \text{inter}_y$$
$$A_1 = (x_2 - x_1 + 1) \times (y_2 - y_1 + 1)$$
$$A_2 = (x_4 - x_3 + 1) \times (y_4 - y_3 + 1)$$
$$\text{IoU} = \frac{A_{\text{inter}}}{A_1 + A_2 - A_{\text{inter}}}$$

---

## 2. VRSBench VQA Protocol

### Matching Rule (from `lx709/VRSBench/eval_fianl/eval_vqa_gpt.ipynb`)
- **Closed-Set Categories:** Questions with target answers in $\{\text{yes}, \text{no}\} \cup \{0, 1, \dots, 99\}$ require **strict exact match**:
  $$\text{Match}(p, g) = \mathbb{I}[\text{clean}(p) = \text{clean}(g)]$$
- **Open-Set Categories:** Cleaned words in ground truth must be a subset of predicted words:
  $$\text{Match}(p, g) = \mathbb{I}[\text{words}(g) \subseteq \text{words}(p)]$$
- **12 Categories Evaluated:** Object category, object existence, object quantity, object color, object shape, object size, object position, object direction, image, scene type, reasoning, rural or urban.

---

## 3. RSVQA-LR Protocol

### Closed-Vocabulary Exact Match Formulation
Unlike the obsolete substring matching (`gt in pred`), the repaired evaluator enforces strict closed-vocabulary classification:
$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[\text{norm}(p_i) = \text{norm}(g_i)]$$
Where:
- For counting questions: `norm()` extracts the integer count; string equality is checked on digits (`int(p) == int(g)`).
- For presence questions: `norm()` maps exclusively to `"yes"` or `"no"`.
- For rural/urban questions: `norm()` maps exclusively to `"rural"` or `"urban"`.

---

## 4. CDVQA Protocol

### Task Formulation
Evaluates bi-temporal change comprehension over sequential image pairs:
$$\text{Input} = [\text{Image}_{T0}, \text{Image}_{T1}, \text{TinyCD\_Overlay}], \quad \text{Prompt} = Q$$
$$\text{Accuracy} = \frac{1}{M} \sum_{j=1}^M \mathbb{I}[\text{norm}(p_j) = \text{norm}(g_j)]$$
Across all 8 official question types.

### Synthetic Reference Prohibition Gate
If any ground truth string matches the legacy synthetic template:
`"New residential buildings and infrastructure constructed in the cleared agricultural area."`
The evaluator raises an immediate `ValueError` and halts execution.

---

## 5. TinyCD (LEVIR-CD) Protocol

Evaluated on 128 pairs at fixed threshold $\tau = 0.50$:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \quad \text{IoU} = \frac{TP}{TP + FP + FN}$$
$$\text{OA} = \frac{TP + TN}{TP + TN + FP + FN}, \quad \text{Specificity} = \frac{TN}{TN + FP}$$

---

## 6. CMAF (WHU-OPT-SAR) Protocol

Evaluated across 4,950 tiles (7 land cover classes, excluding border/void pixels):
$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^C \frac{TP_c}{TP_c + FP_c + FN_c}$$
$$\text{Weighted } F_1 = \sum_{c=1}^C \left( \frac{N_c}{N_{\text{total}}} \right) F_{1, c}$$
