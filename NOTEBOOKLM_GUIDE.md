# 📘 NotebookLM Learning Companion Guide — MicroGPT

## What Is This File?

This file is designed to be uploaded as a **source** into Google NotebookLM alongside the MicroGPT code file (`microgpt.py`). Together, they create an interactive learning environment where NotebookLM acts as your personal tutor.

**How to set up:**
1. Open [NotebookLM](https://notebooklm.google.com/)
2. Create a new notebook
3. Upload `microgpt.py` as Source 1
4. Upload this file (`NOTEBOOKLM_GUIDE.md`) as Source 2
5. Start learning by following the phases below and asking NotebookLM the suggested questions

> **Important for NotebookLM:** When answering questions about concepts in this guide, always reference and quote the relevant lines from the `microgpt.py` source file. Cite specific line numbers and show the actual code. Explain everything in extremely simple, step-by-step language suitable for someone who struggles with math formulas.

---

## 🧭 Learning Rules (For NotebookLM to Follow)

When the learner asks about any concept, NotebookLM should:

1. **Never jump to formulas.** Always explain the plain idea first.
2. **Break everything into tiny steps.** One operation at a time.
3. **Follow this teaching order for every concept:**
   - Plain English explanation
   - Simple real-world analogy
   - Numeric example with small numbers
   - Simple Python code (expanded, no compact syntax)
   - Quote the relevant MicroGPT code and explain each part
4. **Expand all formulas.** Instead of `softmax(QK^T / √d)`, say:
   - Step 1: Compute dot products between query and each key
   - Step 2: Divide each score by the square root of the dimension
   - Step 3: Apply the exponential function to each score
   - Step 4: Divide each result by the total so they sum to 1
5. **Always cite the source file** when referencing code.
6. **Use encouraging, patient language.**

---

# PHASE 1 — Basic Foundations

## Topics to Learn

### 1.1 Numbers in AI
Every piece of text must be converted to numbers before a computer can process it. In MicroGPT, each character (letter) becomes a number called a "token ID."

**Analogy:** A thermometer converts temperature into a number. MicroGPT converts letters into numbers.

**Numeric example:**
```
'a' → 0,  'b' → 1,  'c' → 2,  'e' → 4,  'm' → 12
The word "emma" → [4, 12, 12, 0]
```

**MicroGPT reference:** Look at line 24 (`uchars = sorted(set(...))`) and lines 25-26 for the BOS token and vocab size.

### 1.2 Lists and Vectors
A vector is a list of numbers that together describe something. In MicroGPT, each token is described by a vector of 16 numbers (the "embedding").

**Analogy:** Describe a fruit with numbers: `[sweetness=7, size=5, redness=9]`. Each token gets a similar description, but with 16 numbers instead of 3.

**MicroGPT reference:** Line 75 sets `n_embd = 16`. Line 109 looks up a token's vector: `state_dict['wte'][token_id]`.

### 1.3 Dot Product
The dot product takes two vectors and returns one number measuring their similarity.

**How to compute it (tiny steps):**
```
Vector A: [2, 3, 4]
Vector B: [5, 6, 7]

Step 1: Multiply pairs → 2×5=10, 3×6=18, 4×7=28
Step 2: Add results   → 10 + 18 + 28 = 56
```

**MicroGPT reference:** Line 129 uses dot products in attention: `sum(q_h[j] * k_h[t][j] for j in range(head_dim))`

### 1.4 Matrix Multiplication
A matrix is a list of vectors. Matrix-vector multiplication does one dot product per row.

**How to compute it (tiny steps):**
```
Matrix:              Input vector: [80, 90, 70]
  Row 0: [2, 1, 3]
  Row 1: [1, 4, 1]

Row 0 · input: 2×80 + 1×90 + 3×70 = 160+90+210 = 460
Row 1 · input: 1×80 + 4×90 + 1×70 = 80+360+70 = 510

Output: [460, 510]
```

**MicroGPT reference:** The `linear` function (lines 94-95) does exactly this. It is used everywhere in the model.

### ✅ Phase 1 Checkpoint — Ask NotebookLM:
- [ ] "Show me the `linear` function in microgpt.py and walk through it step by step with a numeric example"
- [ ] "How does MicroGPT convert the letter 'e' into a number? Quote the relevant code."
- [ ] "What is a dot product? Show me where it appears in the attention code."
- [ ] "What does `n_embd = 16` mean and why is it important?"

---

# PHASE 2 — Neural Network Basics

### 2.1 Linear Layers
A linear layer transforms an input vector into an output vector using matrix multiplication. The matrix contains "weights" that the model learns during training.

**MicroGPT reference:** Every call to `linear(x, w)` is a layer. For example, line 138: `linear(x, state_dict[f'layer{li}.mlp_fc1'])` expands the vector from 16 to 64 dimensions.

### 2.2 Activation Functions (ReLU)
ReLU is a simple rule: if a number is positive, keep it; if negative, make it zero.

**Numeric example:**
```
Input:  [-0.5, 0.8, -1.2, 0.3]
ReLU:   [ 0.0, 0.8,  0.0, 0.3]
```

**Why needed:** Without ReLU, stacking many linear layers equals one linear layer. ReLU adds the ability to learn complex patterns.

**MicroGPT reference:** Line 50 defines ReLU: `max(0, self.data)`. Line 139 applies it: `[xi.relu() for xi in x]`.

### 2.3 Softmax
Softmax converts any list of numbers into probabilities that sum to 1.

**Step-by-step with real numbers — input: [2.0, 1.0, 0.5]:**
```
Step 1: Find max           → 2.0
Step 2: Subtract max       → [0.0, -1.0, -1.5]
Step 3: Apply e^x          → [1.000, 0.368, 0.223]
Step 4: Sum them           → 1.591
Step 5: Divide each by sum → [0.629, 0.231, 0.140]
Check: 0.629 + 0.231 + 0.140 = 1.0 ✓
```

**MicroGPT reference:** Lines 97-101 define the `softmax` function.

### ✅ Phase 2 Checkpoint — Ask NotebookLM:
- [ ] "Walk through the softmax function in microgpt.py with the numbers [3.0, 1.0, 0.5]"
- [ ] "Where is ReLU used in the model and why?"
- [ ] "How does the MLP block in microgpt.py transform data? Explain lines 136-141."
- [ ] "What would happen if we removed ReLU from the model?"

---

# PHASE 3 — Learning Mechanism

### 3.1 Loss Function
Loss measures how wrong the model's prediction is. MicroGPT uses cross-entropy: `-log(probability of correct answer)`.

**Intuition:**
```
Model gives correct answer 99% probability → -log(0.99) = 0.01 (tiny loss, great!)
Model gives correct answer 50% probability → -log(0.50) = 0.69 (moderate loss)
Model gives correct answer 1% probability  → -log(0.01) = 4.60 (huge loss, terrible!)
```

**MicroGPT reference:** Line 167: `loss_t = -probs[target_id].log()`

### 3.2 Gradients
A gradient answers: "If I nudge this parameter a tiny bit, does the loss go up or down?"
- Positive gradient → increasing the parameter increases loss (bad!) → decrease it
- Negative gradient → increasing the parameter decreases loss (good!) → increase it

### 3.3 Backpropagation (the chain rule)
The model is a chain of operations. To find how each parameter affects the final loss, multiply the local effects along the chain backward.

**Simple example:**
```
a = 3, b = a × 2 = 6, loss = b + 10 = 16

How does 'a' affect loss?
  loss changes by 1 when b changes by 1 (because loss = b + 10)
  b changes by 2 when a changes by 1 (because b = a × 2)
  Chain rule: 1 × 2 = 2. So loss changes by 2 when a changes by 1.
```

**MicroGPT reference:** The `Value` class (lines 30-72) builds computation graphs automatically. The `backward()` method (lines 59-72) runs backpropagation. Line 72 is the chain rule: `child.grad += local_grad * v.grad`.

### 3.4 Parameter Updates (Adam Optimizer)
After computing gradients, the optimizer adjusts each parameter to reduce the loss.

**MicroGPT reference:** Lines 176-182. Key ideas:
- `m[i]` = momentum (average gradient direction)
- `v[i]` = velocity (average gradient magnitude)
- `p.data -= lr_t * m_hat / (v_hat**0.5 + eps)` = the actual update
- `p.grad = 0` = reset gradient for next step (critical!)

### ✅ Phase 3 Checkpoint — Ask NotebookLM:
- [ ] "Explain the backward() method in the Value class step by step"
- [ ] "What does `child.grad += local_grad * v.grad` mean? Give a numeric trace."
- [ ] "Walk through the Adam optimizer update on lines 176-182 with example numbers"
- [ ] "Why is `p.grad = 0` on line 182 important? What breaks without it?"

---

# PHASE 4 — Transformer Concepts

### 4.1 Token Embeddings
Each token ID is converted into a vector of 16 learned numbers.

**MicroGPT reference:** Line 109: `tok_emb = state_dict['wte'][token_id]` — looks up the embedding for this token.

### 4.2 Position Embeddings
The model also needs to know WHERE a token is in the sequence. Each position (0-15) has its own learned vector.

**MicroGPT reference:** Line 110: `pos_emb = state_dict['wpe'][pos_id]`. Line 111 adds them: `x = [t + p for t, p in zip(tok_emb, pos_emb)]`.

### 4.3 Attention — The Core Innovation
Attention lets each token decide which previous tokens to focus on.

**The Query-Key-Value framework (explained simply):**
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I provide if selected?"

**How attention works (step by step, no compact formulas):**
```
Step 1: Transform current token into Q, K, V vectors (using linear layers)
Step 2: Store K and V so future tokens can reference them
Step 3: Compute similarity score = dot product of Q with each stored K
Step 4: Divide each score by √(head dimension) to keep numbers stable
Step 5: Apply softmax to scores → attention weights (probabilities)
Step 6: Compute weighted sum of all stored V vectors using those weights
Step 7: This weighted sum is the attention output
```

**Numeric example with head_dim=2:**
```
Q = [1.0, 0.5]
K₀ = [0.1, 0.9], K₁ = [0.8, 0.4]
V₀ = [0.2, 0.6], V₁ = [0.9, 0.1]

Dot products:  Q·K₀ = 1.0×0.1 + 0.5×0.9 = 0.55
               Q·K₁ = 1.0×0.8 + 0.5×0.4 = 1.00

Scale (÷√2):   [0.55/1.414, 1.00/1.414] = [0.389, 0.707]
Softmax:        [0.422, 0.578]
Weighted V dim0: 0.422×0.2 + 0.578×0.9 = 0.604
Weighted V dim1: 0.422×0.6 + 0.578×0.1 = 0.311
Output: [0.604, 0.311]
```

**MicroGPT reference:** Lines 118-132 implement multi-head attention. Lines 121-122 store K,V. Line 129 computes scaled dot products. Line 130 applies softmax. Line 131 computes weighted sum.

### 4.4 Multi-Head Attention
Instead of one attention, split into 4 smaller heads (each works on 4 of the 16 dimensions). Each head can learn to look for different patterns.

**MicroGPT reference:** Line 76: `n_head = 4`. Line 79: `head_dim = n_embd // n_head = 4`. Lines 124-132 loop over heads.

### ✅ Phase 4 Checkpoint — Ask NotebookLM:
- [ ] "Walk through the attention computation in microgpt.py lines 118-132 with a numeric example"
- [ ] "What are query, key, and value? Explain using the code on lines 118-120."
- [ ] "Why do we divide by √head_dim on line 129?"
- [ ] "Explain multi-head attention. Why 4 heads instead of 1?"

---

# PHASE 5 — GPT Architecture

### 5.1 Transformer Blocks
Each block has: RMSNorm → Attention → Residual → RMSNorm → MLP → Residual.

### 5.2 Residual Connections
`output = block_result + original_input`. This "skip wire" prevents gradients from vanishing in deep networks.

**MicroGPT reference:** Line 134: `x = [a + b for a, b in zip(x, x_residual)]`

### 5.3 RMSNorm
Normalizes vectors to keep numbers in a stable range.

**Step-by-step for [3.0, 4.0]:**
```
Square each:         9.0, 16.0
Mean of squares:     12.5
Scale = 1/√12.5:    0.283
Multiply:           3.0×0.283=0.849,  4.0×0.283=1.131
Output: [0.849, 1.131]
```

**MicroGPT reference:** Lines 103-106 define `rmsnorm`.

### 5.4 Output Prediction
Final linear layer converts the 16-dim vector to 27 logits (one per possible character).

**MicroGPT reference:** Line 143: `logits = linear(x, state_dict['lm_head'])`

### ✅ Phase 5 Checkpoint — Ask NotebookLM:
- [ ] "Trace the complete flow of one token through the `gpt()` function (lines 108-144)"
- [ ] "What are residual connections and where do they appear in the code?"
- [ ] "Explain `rmsnorm` with a numeric example using the code on lines 103-106"

---

# PHASE 6 — Training Loop

### 6.1 How Training Works
```
Repeat 1000 times:
  1. Pick a name, tokenize it: "emma" → [BOS, 4, 12, 12, 0, BOS]
  2. For each position, predict the next token
  3. Measure loss at each position
  4. Average all losses
  5. Backward pass (compute gradients)
  6. Update parameters (Adam)
  7. Reset gradients
```

**MicroGPT reference:** Lines 153-184 contain the full training loop.

### ✅ Phase 6 Checkpoint — Ask NotebookLM:
- [ ] "Walk through one complete training step using the name 'emma'"
- [ ] "Explain lines 163-169: how are losses computed for each position?"
- [ ] "What does `loss.backward()` on line 172 actually do?"
- [ ] "Why does the learning rate decrease over time? (line 175)"

---

# PHASE 7 — Inference (Generating Text)

### 7.1 How Generation Works
```
Start with BOS → model predicts probabilities → sample a character →
feed it back → repeat → stop when BOS is predicted again
```

### 7.2 Temperature
Dividing logits by temperature before softmax controls randomness:
- Low temperature (0.1): very predictable, safe outputs
- Medium (0.5): balanced creativity (MicroGPT default)
- High (2.0): very random, wild outputs

**MicroGPT reference:** Lines 186-200. Line 195 applies temperature: `softmax([l / temperature for l in logits])`.

### ✅ Phase 7 Checkpoint — Ask NotebookLM:
- [ ] "Walk through generating one name character by character using the inference code"
- [ ] "What does temperature do? Show what happens with temp=0.1 vs temp=2.0"
- [ ] "What is `random.choices` doing on line 196?"

---

# SECTION A — How to Ask NotebookLM Questions

Use these question patterns for the best results:

| Pattern | Example |
|---|---|
| **Explain code** | "Explain lines 59-72 of microgpt.py in simple terms" |
| **Numeric trace** | "Walk through the softmax function with input [3.0, 1.0, 0.5]" |
| **Why questions** | "Why does line 72 use += instead of =?" |
| **What-if** | "What would happen if we removed RMSNorm from the model?" |
| **Connect concepts** | "How does the backward() method relate to the training loop?" |
| **Compare** | "What is the difference between token embedding and position embedding?" |
| **Simplify** | "Explain line 129 like I'm 12 years old" |

---

# SECTION B — Prompts to Explore the Code File

Copy-paste these into NotebookLM to explore the code:

**Understanding structure:**
- "List all the functions defined in microgpt.py and explain what each one does in one sentence"
- "What are all the hyperparameters in microgpt.py? List them with their values and meanings"
- "Show me the complete flow: what happens from when a name enters the model to when a loss is computed?"

**Deep dives:**
- "Walk through the Value class. What does each method do? Give a simple example for each."
- "Explain the `gpt()` function line by line. What goes in and what comes out?"
- "How does the Adam optimizer on lines 176-182 differ from a simple `param -= lr * grad`?"

**Connections:**
- "Where is the `linear()` function called in the code? List every usage and explain why."
- "Trace how a single parameter's gradient flows from the loss back through the network"
- "How do `keys` and `values` lists grow during training? Why is this needed?"

---

# SECTION C — Debugging and Experimentation Prompts

Ask NotebookLM these to deepen understanding through hypotheticals:

- "What would break if I removed `p.grad = 0` on line 182?"
- "What would happen if I set `n_head = 1` instead of 4?"
- "What if I removed all residual connections (the `x = [a + b ...]` lines)?"
- "What happens if learning_rate is set to 100? Why?"
- "What would change if I used 2 layers instead of 1 (`n_layer = 2`)?"
- "Why does the model use BOS as both the start AND end token?"
- "What if I removed the `/ head_dim**0.5` scaling from line 129?"

---

# SECTION D — Reflection Questions to Test Understanding

After each phase, test yourself by answering these WITHOUT looking at the code. Then verify with NotebookLM.

**Phase 1-2 Reflection:**
1. What is a dot product and why does attention use it?
2. What does the `linear()` function do in one sentence?
3. Why do we need ReLU between linear layers?
4. What does softmax guarantee about its output?

**Phase 3 Reflection:**
5. What does the loss measure? What makes it go up or down?
6. What is a gradient, in one sentence?
7. Why do we walk BACKWARD through the graph, not forward?
8. Why must gradients be reset to zero after each training step?

**Phase 4-5 Reflection:**
9. What is the purpose of Query, Key, and Value in attention?
10. Why divide attention scores by √(head_dim)?
11. What is a residual connection and why is it important?
12. What does RMSNorm prevent?

**Phase 6-7 Reflection:**
13. What are the 5 main steps of one training iteration?
14. How does the model generate text one character at a time?
15. What does temperature control during inference?

> **Ask NotebookLM:** "Check my answer to question [N]: [your answer]. Is it correct? What am I missing?"

---

# 🗺️ Quick Reference — Code Map

| Lines | What | Phase |
|---|---|---|
| 9-12 | Imports, random seed | Setup |
| 14-21 | Load dataset | 1 |
| 23-27 | Tokenizer | 1 |
| 30-72 | **Value class (autograd)** | 3 |
| 74-90 | Model parameters | 1-2 |
| 94-95 | `linear()` | 1 |
| 97-101 | `softmax()` | 2 |
| 103-106 | `rmsnorm()` | 5 |
| 108-144 | **`gpt()` forward pass** | 4-5 |
| 146-149 | Optimizer buffers | 3 |
| 151-184 | **Training loop** | 6 |
| 186-200 | **Inference** | 7 |

---

*Upload this file alongside `microgpt.py` in NotebookLM and start with Phase 1. Take your time. Learning to build a GPT is a marathon, not a sprint.* 🚀
