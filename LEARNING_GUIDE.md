# 🧠 From Zero to GPT: A Step-by-Step Learning Guide

### *For people who struggle with formulas. Every concept is broken into the tiniest possible steps.*

> This guide uses [microgpt.py](./microgpt.py) — a 200-line GPT written in pure Python — as your anchor.
> You do NOT need to know math or neural networks. We start from absolute zero.

---

## How This Guide Works

We build everything from the simplest ideas up:

```
Step 1:  Numbers
Step 2:  Lists
Step 3:  Vectors
Step 4:  Dot Product
Step 5:  Matrix Multiplication
Step 6:  Neural Network Layers
Step 7:  Softmax (turning numbers into probabilities)
Step 8:  Loss (measuring how wrong the model is)
Step 9:  Gradients and Backpropagation (teaching the model)
Step 10: Attention (the core idea of transformers)
Step 11: The Full Transformer / GPT
Step 12: Training Loop
Step 13: Inference (generating text)
```

**Every single concept follows this pattern:**
1. 💬 Simple idea in plain language
2. 🏠 Real-world analogy
3. 🧮 Smallest logical steps with real numbers
4. 🐍 Very simple Python example
5. 📌 How it appears in the MicroGPT code

---

# PART 1: Building Blocks (The Foundation)

---

## Step 1: Numbers

### 💬 Simple idea
Everything in a computer is a number. Text, images, sounds — everything gets converted to numbers before a computer can work with it.

### 🏠 Analogy
Think of a thermometer. It converts "how hot is it?" into a number: 72°F. The computer does the same thing with words — it converts "how meaningful is this word?" into numbers.

### 🧮 Tiny steps
```
The letter "a" might become the number 0
The letter "b" might become the number 1
The letter "c" might become the number 2
```

### 🐍 Simple Python
```python
# We have a letter
letter = "a"

# We convert it to a number
number = 0

print(number)  # Output: 0
```

### 📌 In MicroGPT (line 24)
```python
uchars = sorted(set(''.join(docs)))
# This creates a list of all unique characters
# Each character's position in the list becomes its number (token ID)
# For example: uchars = ['a', 'b', 'c', ...] → 'a'=0, 'b'=1, 'c'=2, ...
```

---

## Step 2: Lists

### 💬 Simple idea
A list is just a collection of numbers put together in a row.

### 🏠 Analogy
Think of a shopping list. Instead of items, we have numbers:
```
Shopping list: [eggs, milk, bread]
Number list:   [3, 7, 2]
```

### 🧮 Tiny steps
```
A single number:        5
A list of numbers:      [5, 3, 8]
A list has a length:    this list has 3 items
You can access items:   the first item is 5, the second is 3
```

### 🐍 Simple Python
```python
# Create a list
my_list = [5, 3, 8]

# How many items?
length = len(my_list)
print(length)         # Output: 3

# Get the first item (counting starts from 0)
first = my_list[0]
print(first)          # Output: 5

# Get the second item
second = my_list[1]
print(second)         # Output: 3
```

---

## Step 3: Vectors

### 💬 Simple idea
A vector is just a list of numbers that together describe something. That's it. There's nothing more magical to it.

### 🏠 Analogy
Imagine you describe a fruit with 3 numbers:
```
[sweetness, size, redness]

Apple:  [7, 5, 9]    → sweet, medium, very red
Lemon:  [2, 3, 1]    → not sweet, small, not red
```

Each fruit is different, and the list of numbers captures those differences.

In a GPT, each word/letter gets described by a list of numbers (a vector). The model *learns* what these numbers should be.

### 🧮 Tiny steps
```
A vector is a list of numbers:       [0.5, -0.3, 0.8]
This vector has 3 dimensions.
Each number is one "feature" or "aspect" of what we're describing.

In MicroGPT, each character is described by 16 numbers.
So the vector for "a" might look like: [0.12, -0.45, 0.78, 0.01, ..., 0.33]  (16 numbers)
```

### 🐍 Simple Python
```python
# A vector describing the letter "a"
vector_a = [0.5, -0.3, 0.8]

# A vector describing the letter "b"
vector_b = [0.1, 0.7, -0.2]

# They are just lists of numbers!
print(len(vector_a))  # Output: 3 (this vector has 3 dimensions)
```

### 📌 In MicroGPT (lines 75, 109)
```python
n_embd = 16  # each token is described by 16 numbers

# The embedding table: one vector (list of 16 numbers) for each token
# state_dict['wte'] is a big list of vectors
# state_dict['wte'][token_id] gives you the vector for that token
tok_emb = state_dict['wte'][token_id]
# tok_emb is now a list of 16 numbers describing this token
```

---

## Step 4: Dot Product

### 💬 Simple idea
The dot product tells you **how similar** two vectors are. It takes two lists of numbers and produces a single number.

### 🏠 Analogy
Imagine two students rating 3 movies on a scale of 1-10:
```
Student A ratings: [9, 2, 8]   (loves action & comedy, hates romance)
Student B ratings: [8, 3, 7]   (similar taste!)
Student C ratings: [1, 9, 2]   (opposite taste!)
```

The dot product between A and B will be HIGH (similar tastes).
The dot product between A and C will be LOW (different tastes).

### 🧮 Tiny steps — how to compute a dot product

Take two vectors: `[2, 3, 4]` and `[5, 6, 7]`

```
Step 1: Pair up the numbers at each position
   Position 0:  2 and 5
   Position 1:  3 and 6
   Position 2:  4 and 7

Step 2: Multiply each pair
   2 × 5 = 10
   3 × 6 = 18
   4 × 7 = 28

Step 3: Add all the results
   10 + 18 + 28 = 56

The dot product is 56.
```

### 🐍 Simple Python (expanded, step-by-step)
```python
vector_1 = [2, 3, 4]
vector_2 = [5, 6, 7]

# Step 1 and 2: Multiply each pair
product_0 = vector_1[0] * vector_2[0]   # 2 × 5 = 10
product_1 = vector_1[1] * vector_2[1]   # 3 × 6 = 18
product_2 = vector_1[2] * vector_2[2]   # 4 × 7 = 28

# Step 3: Add them all up
result = product_0 + product_1 + product_2   # 10 + 18 + 28 = 56

print(result)  # Output: 56
```

Now let's write it with a loop (so it works for any length):
```python
vector_1 = [2, 3, 4]
vector_2 = [5, 6, 7]

total = 0
for i in range(len(vector_1)):
    pair_product = vector_1[i] * vector_2[i]
    total = total + pair_product

print(total)  # Output: 56
```

### 📌 In MicroGPT (line 129)
The dot product appears in the attention mechanism:
```python
# This computes the dot product between query and key vectors
# q_h = query vector for one attention head
# k_h[t] = key vector for position t

# The compact version in MicroGPT:
# sum(q_h[j] * k_h[t][j] for j in range(head_dim))

# Expanded, that means:
total = 0
for j in range(head_dim):    # for each dimension
    product = q_h[j] * k_h[t][j]   # multiply the pair
    total = total + product          # add to running total
# total is now the dot product — how much this query "matches" this key
```

---

## Step 5: Matrix Multiplication (one row at a time)

### 💬 Simple idea
A matrix is a list of vectors (a list of lists). Matrix multiplication is just doing **many dot products** — one for each row.

### 🏠 Analogy
Imagine you have 3 teachers, each grading a student on different criteria:
```
Teacher 1 weights: [2, 1, 3]   (values math, english, science differently)
Teacher 2 weights: [1, 4, 1]
Teacher 3 weights: [3, 2, 0]
```
The student's scores are: `[80, 90, 70]`

Each teacher computes their own weighted score (dot product with student scores).
You get one final score per teacher. That's matrix-vector multiplication!

### 🧮 Tiny steps

Matrix (3 rows, each row is a vector):
```
Row 0: [2, 1, 3]
Row 1: [1, 4, 1]
Row 2: [3, 2, 0]
```

Input vector: `[80, 90, 70]`

```
Step 1: Dot product of Row 0 with input
   2×80 + 1×90 + 3×70 = 160 + 90 + 210 = 460

Step 2: Dot product of Row 1 with input
   1×80 + 4×90 + 1×70 = 80 + 360 + 70 = 510

Step 3: Dot product of Row 2 with input
   3×80 + 2×90 + 0×70 = 240 + 180 + 0 = 420

Output: [460, 510, 420]
```

One input vector → one output vector. Each row of the matrix produces one number in the output.

### 🐍 Simple Python (fully expanded)
```python
# The matrix (a list of rows)
w = [
    [2, 1, 3],   # row 0
    [1, 4, 1],   # row 1
    [3, 2, 0],   # row 2
]

# The input vector
x = [80, 90, 70]

# Process each row
output = []

for row in w:
    # Compute dot product of this row with input
    total = 0
    for wi, xi in zip(row, x):
        total = total + (wi * xi)
    output.append(total)

print(output)  # [460, 510, 420]
```

### 📌 In MicroGPT (lines 94-106 in Scratch version)
This is exactly the `linear` function:
```python
def linear(x, w):
    output = []
    for row in w:          # for each row in the weight matrix
        total = 0
        for wi, xi in zip(row, x):   # pair up and multiply
            total += wi * xi
        output.append(total)   # one output per row
    return output
```

This function is used EVERYWHERE in the model:
- Converting tokens to queries, keys, values (attention)
- The MLP layers
- The final prediction layer

---

## Step 6: Neural Network Layers

### 💬 Simple idea
A neural network layer takes an input vector and transforms it into a new vector. That's it. The transformation is the `linear` function we just learned — matrix multiplication.

### 🏠 Analogy
Think of a translator. You speak a sentence in English (input vector). The translator converts it to French (output vector). The translator's "knowledge" is stored in the weights (the matrix).

A neural network has many layers — like a chain of translators. English → French → German → Spanish. Each layer transforms the information into a new form.

### 🧮 Tiny steps
```
Input:   [0.5, -0.3, 0.8]     (3 numbers in)

Weights (a 2×3 matrix):
  Row 0: [0.1, 0.4, -0.2]
  Row 1: [0.3, -0.1, 0.5]

Output:
  Row 0 dot input: 0.1×0.5 + 0.4×(-0.3) + (-0.2)×0.8 = 0.05 - 0.12 - 0.16 = -0.23
  Row 1 dot input: 0.3×0.5 + (-0.1)×(-0.3) + 0.5×0.8 = 0.15 + 0.03 + 0.40 = 0.58

Output: [-0.23, 0.58]         (2 numbers out)
```

The matrix has 2 rows → 2 outputs. The matrix has 3 columns → accepts 3 inputs.
**The weights decide what transformation happens.** Training is about finding the right weights.

### Why do we need non-linearity (ReLU)?

Without ReLU, stacking many layers is the same as having one layer. ReLU adds "choices" — it says:

```
If the number is positive → keep it
If the number is negative → change it to 0
```

### 🧮 ReLU step by step
```
Input:  [-0.23, 0.58, -1.5, 0.3]

Apply ReLU to each number:
  -0.23 → negative → becomes 0
   0.58 → positive → stays 0.58
  -1.5  → negative → becomes 0
   0.3  → positive → stays 0.3

Output: [0, 0.58, 0, 0.3]
```

### 🐍 Simple Python
```python
def relu(x):
    if x > 0:
        return x
    else:
        return 0

print(relu(0.58))   # Output: 0.58 (positive, kept)
print(relu(-0.23))  # Output: 0    (negative, zeroed)
```

### 📌 In MicroGPT (lines 50, 139)
```python
# In the Value class (line 50):
def relu(self):
    return Value(max(0, self.data), ...)

# In the MLP block (line 139):
x = [xi.relu() for xi in x]
# This applies relu to every number in the vector:
#   for each xi in x:
#       if xi > 0: keep it
#       if xi <= 0: make it 0
```

---

## Step 7: Softmax — Turning Numbers into Probabilities

### 💬 Simple idea
Softmax takes a list of numbers (any numbers — big, small, negative) and converts them into **probabilities** that add up to 1.

### 🏠 Analogy
Three restaurants have these ratings: `[5.0, 3.0, 1.0]`

You want to turn these into chances of which one you'd pick:
- Rating 5.0 → high chance
- Rating 3.0 → medium chance
- Rating 1.0 → low chance

Softmax does exactly this — higher numbers get higher probabilities.

### 🧮 Tiny steps

Input numbers (called "logits"): `[2.0, 1.0, 0.5]`

```
Step 1: Find the biggest number
   max_val = 2.0

Step 2: Subtract the max from each number (for numerical stability)
   2.0 - 2.0 = 0.0
   1.0 - 2.0 = -1.0
   0.5 - 2.0 = -1.5

Step 3: Apply the exponential function (e^x) to each
   e^(0.0)  = 1.000
   e^(-1.0) = 0.368
   e^(-1.5) = 0.223

   Why e^x? Because:
   - It makes all numbers positive
   - It amplifies differences (bigger numbers get MUCH bigger)
   - It's smooth and mathematically nice for gradients

Step 4: Add them all up
   total = 1.000 + 0.368 + 0.223 = 1.591

Step 5: Divide each by the total
   1.000 / 1.591 = 0.629
   0.368 / 1.591 = 0.231
   0.223 / 1.591 = 0.140

Output probabilities: [0.629, 0.231, 0.140]
Check: 0.629 + 0.231 + 0.140 = 1.0 ✓
```

The biggest input (2.0) got the biggest probability (0.629). ✓

### 🐍 Simple Python (fully expanded)
```python
import math

logits = [2.0, 1.0, 0.5]

# Step 1: Find the max
max_val = max(logits)   # 2.0

# Step 2: Subtract max from each
shifted = []
for val in logits:
    shifted.append(val - max_val)
# shifted = [0.0, -1.0, -1.5]

# Step 3: Apply e^x to each
exps = []
for val in shifted:
    exps.append(math.exp(val))
# exps = [1.000, 0.368, 0.223]

# Step 4: Sum them
total = 0
for val in exps:
    total = total + val
# total = 1.591

# Step 5: Divide each by total
probs = []
for val in exps:
    probs.append(val / total)
# probs = [0.629, 0.231, 0.140]

print(probs)
```

### 📌 In MicroGPT (lines 110-114)
```python
def softmax(logits):
    max_val = max(val.data for val in logits)         # Step 1: find max
    exps = [(val - max_val).exp() for val in logits]  # Steps 2+3: subtract & exponentiate
    total = sum(exps)                                  # Step 4: sum
    return [e / total for e in exps]                   # Step 5: divide

# It's the same 5 steps, just written more compactly!
```

---

## Step 8: Loss — Measuring How Wrong the Model Is

### 💬 Simple idea
After the model makes a prediction (probabilities for each token), we need to measure: **how good or bad was that prediction?** That measurement is called the **loss**.

### 🏠 Analogy
A student takes a test. The loss is like the score:
- If the student is very confident in the RIGHT answer → **low** loss (good!)
- If the student is very confident in the WRONG answer → **high** loss (bad!)
- If the student has no idea → **medium** loss

### 🧮 Tiny steps — Cross-Entropy Loss

The model outputs probabilities for every possible next character. We know which character actually comes next (the "target"). We look at the probability the model assigned to the correct answer.

```
Model's probabilities: [0.05, 0.10, 0.70, 0.15]
                        "a"    "b"    "c"    "d"

The correct answer is: "c" (index 2)

The model gave "c" a probability of 0.70

Loss = -log(probability of correct answer)
Loss = -log(0.70)
```

Now let's understand `-log(probability)`:

```
If probability is VERY HIGH (model is right):
   -log(0.99) = 0.01    → tiny loss! ✓

If probability is MEDIUM (model is unsure):
   -log(0.50) = 0.69    → moderate loss

If probability is VERY LOW (model is wrong):
   -log(0.01) = 4.60    → HUGE loss! ✗
```

The pattern: higher probability → lower loss. The model is punished more when it's wrong.

### 🐍 Simple Python (fully expanded)
```python
import math

# Model's predicted probabilities for each character
probs = [0.05, 0.10, 0.70, 0.15]

# The correct character is "c", which is at index 2
target_index = 2

# Step 1: Look up the probability of the correct answer
prob_of_correct = probs[target_index]   # 0.70

# Step 2: Take the logarithm
log_prob = math.log(prob_of_correct)     # log(0.70) = -0.357

# Step 3: Negate it (because log of a probability is always negative)
loss = -log_prob                          # -(-0.357) = 0.357

print(loss)  # 0.357 — not too bad, model was fairly confident
```

### 📌 In MicroGPT (lines 167-169)
```python
probs = softmax(logits)            # get probabilities from model output
loss_t = -probs[target_id].log()   # look up correct answer's prob, take -log

# This is the same as our expanded version:
# prob_of_correct = probs[target_id]
# log_prob = prob_of_correct.log()
# loss_t = -log_prob
```

---

## Step 9: Gradients and Backpropagation — Teaching the Model

### 💬 Simple idea
A gradient answers one question: **"If I change this number a tiny bit, does the loss go up or down?"**

- If the gradient is **positive** → increasing this number makes loss go UP (bad!)
- If the gradient is **negative** → increasing this number makes loss go DOWN (good!)
- If the gradient is **zero** → changing this number doesn't affect the loss

### 🏠 Analogy
Imagine you're standing on a hill and you want to reach the valley (lowest point = lowest loss).

- The gradient is like looking at the slope under your feet.
- If the slope goes uphill to the right → step LEFT (opposite direction!).
- If the slope goes uphill to the left → step RIGHT.
- The steeper the slope, the bigger step you should take.

### 🧮 Tiny steps — the chain rule

The model is a chain of operations:

```
input → [operation A] → [operation B] → [operation C] → loss
```

We want to know: how does changing the input affect the loss?

The chain rule says: **multiply the effects along the chain.**

```
Example:
  a = 3
  b = a × 2         (operation: multiply by 2)
  c = b + 10         (operation: add 10)
  loss = c × c       (operation: square)

Forward pass (compute values):
  a = 3
  b = 3 × 2 = 6
  c = 6 + 10 = 16
  loss = 16 × 16 = 256

Backward pass (compute gradients):
  How does loss change when c changes?
    loss = c × c, so the gradient is 2 × c = 2 × 16 = 32
    (If c goes up by 1, loss goes up by about 32)

  How does c change when b changes?
    c = b + 10, so the gradient is 1
    (If b goes up by 1, c goes up by 1)

  How does b change when a changes?
    b = a × 2, so the gradient is 2
    (If a goes up by 1, b goes up by 2)

  Chain rule: how does loss change when a changes?
    Multiply all the gradients together: 32 × 1 × 2 = 64
    (If a goes up by 1, loss goes up by about 64)
```

### 🐍 Simple Python — tracing a computation graph
```python
# Let's trace through MicroGPT's Value class manually

# Step 1: Create values
a = 3.0
b = 2.0

# Step 2: Forward pass — compute the result
c = a * b        # c = 6.0
d = c + 10       # d = 16.0
loss = d * d     # loss = 256.0

# Step 3: Backward pass — compute gradients
# Start from the end: dloss/dloss = 1 (always!)
grad_loss = 1.0

# loss = d * d → gradient of d is: 2 * d * grad_loss
grad_d = 2 * d * grad_loss          # 2 × 16 × 1 = 32

# d = c + 10 → gradient of c is: 1 * grad_d  (adding doesn't change the gradient)
grad_c = 1 * grad_d                  # 1 × 32 = 32

# c = a * b → gradient of a is: b * grad_c (the OTHER input's value)
grad_a = b * grad_c                  # 2 × 32 = 64

# c = a * b → gradient of b is: a * grad_c
grad_b = a * grad_c                  # 3 × 32 = 96

print(f"grad_a = {grad_a}")  # 64
print(f"grad_b = {grad_b}")  # 96
```

### Understanding the Value class

The `Value` class in MicroGPT does all of this automatically. When you write `a + b`, it:
1. Computes the result (`data`)
2. Remembers the inputs (`children`)
3. Records how to compute gradients (`local_grads`)

Let's trace through an addition:

```python
# When you write:
a = Value(3.0)
b = Value(5.0)
c = a + b

# Behind the scenes, Python calls a.__add__(b), which does:
#   result_data = 3.0 + 5.0 = 8.0
#   children = (a, b)
#   local_grads = (1, 1)   ← because d(a+b)/da = 1 and d(a+b)/db = 1
#   c = Value(8.0, children=(a, b), local_grads=(1, 1))
```

And for multiplication:
```python
# When you write:
c = a * b

# Behind the scenes:
#   result_data = 3.0 × 5.0 = 15.0
#   children = (a, b)
#   local_grads = (b.data, a.data) = (5.0, 3.0)
#   ← because d(a×b)/da = b and d(a×b)/db = a
```

### The backward() function — step by step

```
Step 1: Find the right order to process nodes
   We need to process nodes "backwards" — loss first, inputs last.
   This is called "topological sort."

Step 2: Set the starting gradient
   The gradient of the loss with respect to itself is always 1.
   (If we increase the loss by 1, the loss increases by 1. Obviously!)

Step 3: Walk backwards through each node
   For each node, tell its children:
   "Hey, your gradient is my gradient × the local gradient"

   This += is important because a node might receive gradients from
   multiple paths. We ADD them all up.
```

### 📌 In MicroGPT (lines 59-72)
```python
def backward(self):
    # Step 1: Build the processing order (topological sort)
    topo = []             # will hold nodes in the right order
    visited = set()       # track which nodes we've already seen

    def build_topo(v):
        if v not in visited:        # haven't processed this node yet?
            visited.add(v)          # mark it as seen
            for child in v._children:
                build_topo(child)   # process children first (recursion!)
            topo.append(v)          # then add this node

    build_topo(self)   # start from the loss node

    # Step 2: Set the starting gradient
    self.grad = 1      # dloss/dloss = 1

    # Step 3: Walk backwards
    for v in reversed(topo):    # process in reverse order
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
            # ↑ THIS IS THE CHAIN RULE!
            # child's gradient += (how much child affects v) × (how much v affects loss)
```

### How parameters get updated (the optimizer)

Once we know the gradient of each parameter, we "nudge" the parameter in the **opposite** direction of the gradient:

```
If gradient is positive (increasing param makes loss go UP):
   → decrease the parameter

If gradient is negative (increasing param makes loss go DOWN):
   → increase the parameter
```

The simplest version:
```python
parameter = parameter - learning_rate * gradient
```

### 📌 In MicroGPT (lines 176-182) — Adam optimizer, step by step
```python
# Adam is a smarter version of the simple update above.
# It keeps track of:
#   m[i] = the "momentum" — average direction of recent gradients
#   v[i] = the "velocity" — average SIZE of recent gradients

for i, p in enumerate(params):
    # Update momentum (which direction have gradients been going?)
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
    # Translation: m is 85% old momentum + 15% new gradient

    # Update velocity (how BIG have gradients been?)
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
    # Translation: v is 99% old velocity + 1% new gradient squared

    # Bias correction (fix the initial underestimate)
    m_hat = m[i] / (1 - beta1 ** (step + 1))
    v_hat = v[i] / (1 - beta2 ** (step + 1))

    # Finally, update the parameter!
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
    # Translation: step in the direction of momentum,
    # but take smaller steps when velocity is high (gradient is noisy)

    p.grad = 0  # CRITICAL: reset gradient for next step
```

---

# PART 2: The Transformer / GPT

---

## Step 10: Attention — The Core Idea

### 💬 Simple idea
When you read a sentence, you don't pay equal attention to every word. Some words are more important for understanding the current word. **Attention** lets the model decide which previous words to focus on.

### 🏠 Analogy
You're at a party and someone says: *"The cat sat on the ___"*

Your brain automatically focuses on "cat" and "sat" to predict the next word (probably "mat" or "chair"). You mostly ignore "the" and "on" — they're less helpful.

Attention does the same thing — for each position, it computes: **"How much should I focus on each previous position?"**

### The Query-Key-Value idea (explained simply)

Every word/token gets converted into three different vectors:

```
Query (Q):  "What am I LOOKING FOR?"
Key (K):    "What do I CONTAIN?"
Value (V):  "What INFORMATION do I provide?"
```

Think of it like a library:
- **Query** = "I'm looking for books about cooking"
- **Key** = the label on each book ("cooking", "history", "science", ...)
- **Value** = the actual content inside the book

The process:
1. Compare my Query to every book's Key (dot product → similarity score)
2. Books with matching Keys get high scores
3. Convert scores to probabilities (softmax)
4. Read a weighted mix of the book Values

### 🧮 Tiny steps — computing attention for one position

Let's say we have 3 positions processed so far, with `head_dim=2`:

```
Current query:     Q = [1.0, 0.5]    ("What am I looking for?")

Keys of past positions:
  K₀ = [0.1, 0.9]    (key at position 0)
  K₁ = [0.8, 0.4]    (key at position 1)
  K₂ = [0.3, 0.7]    (key at position 2)

Values of past positions:
  V₀ = [0.2, 0.6]    (info at position 0)
  V₁ = [0.9, 0.1]    (info at position 1)
  V₂ = [0.5, 0.5]    (info at position 2)
```

**Step 1: Compute dot products (Q · each K)**
```
  Q · K₀ = (1.0 × 0.1) + (0.5 × 0.9) = 0.10 + 0.45 = 0.55
  Q · K₁ = (1.0 × 0.8) + (0.5 × 0.4) = 0.80 + 0.20 = 1.00
  Q · K₂ = (1.0 × 0.3) + (0.5 × 0.7) = 0.30 + 0.35 = 0.65

  Scores: [0.55, 1.00, 0.65]
  K₁ has the highest score — it matches the query best!
```

**Step 2: Scale by dividing by √head_dim**
```
  √2 ≈ 1.414

  0.55 / 1.414 = 0.389
  1.00 / 1.414 = 0.707
  0.65 / 1.414 = 0.460

  Scaled scores: [0.389, 0.707, 0.460]
```

Why scale? Without scaling, when `head_dim` is large, dot products become very large numbers, and softmax becomes too "peaked" (one position gets almost all the weight).

**Step 3: Softmax → convert to probabilities**
```
  (Using the softmax steps from Step 7 above)

  max_val = 0.707
  Subtract max:    [0.389-0.707, 0.707-0.707, 0.460-0.707] = [-0.318, 0.000, -0.247]
  Exponentiate:    [e^(-0.318), e^(0.000), e^(-0.247)] = [0.728, 1.000, 0.781]
  Sum:             0.728 + 1.000 + 0.781 = 2.509
  Divide:          [0.728/2.509, 1.000/2.509, 0.781/2.509]
                 = [0.290, 0.399, 0.311]

  Attention weights: [0.290, 0.399, 0.311]
  Position 1 gets the most attention (0.399). ✓
```

**Step 4: Weighted sum of values**
```
  For each dimension of the output:

  Dimension 0:
    0.290 × V₀[0] + 0.399 × V₁[0] + 0.311 × V₂[0]
    = 0.290 × 0.2 + 0.399 × 0.9 + 0.311 × 0.5
    = 0.058 + 0.359 + 0.156
    = 0.573

  Dimension 1:
    0.290 × V₀[1] + 0.399 × V₁[1] + 0.311 × V₂[1]
    = 0.290 × 0.6 + 0.399 × 0.1 + 0.311 × 0.5
    = 0.174 + 0.040 + 0.156
    = 0.370

  Attention output: [0.573, 0.370]
```

This output is a **blend** of all the values, weighted by how relevant each position is.

### 🐍 Simple Python (fully expanded)
```python
import math

# Our vectors (head_dim = 2)
Q = [1.0, 0.5]
keys = [[0.1, 0.9], [0.8, 0.4], [0.3, 0.7]]
vals = [[0.2, 0.6], [0.9, 0.1], [0.5, 0.5]]
head_dim = 2

# --- Step 1: Dot products ---
scores = []
for t in range(len(keys)):
    dot = 0
    for j in range(head_dim):
        dot = dot + Q[j] * keys[t][j]
    scores.append(dot)
# scores = [0.55, 1.00, 0.65]

# --- Step 2: Scale ---
scale = head_dim ** 0.5   # sqrt(2) ≈ 1.414
for t in range(len(scores)):
    scores[t] = scores[t] / scale
# scores = [0.389, 0.707, 0.460]

# --- Step 3: Softmax ---
max_val = max(scores)
exps = []
for s in scores:
    exps.append(math.exp(s - max_val))
total = sum(exps)
attn_weights = []
for e in exps:
    attn_weights.append(e / total)
# attn_weights = [0.290, 0.399, 0.311]

# --- Step 4: Weighted sum of values ---
output = []
for j in range(head_dim):       # for each dimension of the output
    weighted_sum = 0
    for t in range(len(vals)):   # for each position
        weighted_sum = weighted_sum + attn_weights[t] * vals[t][j]
    output.append(weighted_sum)
# output = [0.573, 0.370]

print("Attention weights:", attn_weights)
print("Output:", output)
```

### Multi-Head Attention — doing this multiple times in parallel

Instead of one big attention computation, we split the embedding into **multiple heads**, each working on a smaller slice:

```
Full embedding: 16 numbers
4 heads, each gets: 16 / 4 = 4 numbers

Head 0 works on dimensions [0, 1, 2, 3]
Head 1 works on dimensions [4, 5, 6, 7]
Head 2 works on dimensions [8, 9, 10, 11]
Head 3 works on dimensions [12, 13, 14, 15]
```

Why? Each head can learn to look for different patterns!
- Head 0 might look for "what letter comes before me?"
- Head 1 might look for "is there a vowel nearby?"
- Head 2 might look for "what position am I at?"
- Head 3 might look for something else entirely

### 📌 In MicroGPT (lines 118-132)
```python
# Step 1: Create Q, K, V using linear transformations
q = linear(x, state_dict[f'layer{li}.attn_wq'])   # query: "what am I looking for?"
k = linear(x, state_dict[f'layer{li}.attn_wk'])   # key: "what do I contain?"
v = linear(x, state_dict[f'layer{li}.attn_wv'])   # value: "what info do I provide?"

# Save K and V for future positions to use
keys[li].append(k)
values[li].append(v)

# For each head:
x_attn = []
for h in range(n_head):
    hs = h * head_dim     # starting index for this head's slice

    # Slice out this head's portion
    q_h = q[hs:hs+head_dim]                         # this head's query
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]   # this head's keys (all positions)
    v_h = [vi[hs:hs+head_dim] for vi in values[li]] # this head's values (all positions)

    # Compute attention scores (dot product, scaled)
    attn_logits = []
    for t in range(len(k_h)):
        dot = 0
        for j in range(head_dim):
            dot = dot + q_h[j] * k_h[t][j]
        attn_logits.append(dot / head_dim**0.5)

    # Softmax → probabilities
    attn_weights = softmax(attn_logits)

    # Weighted sum of values
    head_out = []
    for j in range(head_dim):
        weighted_sum = 0
        for t in range(len(v_h)):
            weighted_sum = weighted_sum + attn_weights[t] * v_h[t][j]
        head_out.append(weighted_sum)

    # Concatenate this head's output
    x_attn.extend(head_out)

# All heads concatenated back into a full-size vector
```

---

## Step 11: RMSNorm — Keeping Numbers Stable

### 💬 Simple idea
When numbers pass through many operations, they can become extremely large or extremely small. Normalization rescales them to a reasonable range.

### 🏠 Analogy
Imagine you're adjusting the volume on different songs. Some songs are recorded very loud, others very quiet. Normalization is like an auto-volume feature — it brings everything to a similar loudness.

### 🧮 Tiny steps

Input vector: `[3.0, 4.0]`

```
Step 1: Square each number
   3.0² = 9.0
   4.0² = 16.0

Step 2: Find the average of the squares (mean square)
   (9.0 + 16.0) / 2 = 12.5

Step 3: Take the square root
   √12.5 = 3.536

Step 4: Divide each original number by this
   3.0 / 3.536 = 0.849
   4.0 / 3.536 = 1.131

Output: [0.849, 1.131]
```

The numbers are now in a manageable range, but their relative sizes are preserved (4.0 was bigger than 3.0 before, and 1.131 is bigger than 0.849 after).

### 🐍 Simple Python (fully expanded)
```python
x = [3.0, 4.0]

# Step 1: Square each number
squares = []
for xi in x:
    squares.append(xi * xi)
# squares = [9.0, 16.0]

# Step 2: Average of squares
mean_square = sum(squares) / len(squares)   # 12.5

# Step 3: Compute the scaling factor (1 / sqrt(mean_square))
# We add a tiny number (1e-5) to avoid dividing by zero
scale = (mean_square + 0.00001) ** (-0.5)    # 1/√12.5 ≈ 0.283

# Step 4: Multiply each number by the scale
output = []
for xi in x:
    output.append(xi * scale)
# output = [0.849, 1.131]

print(output)
```

### 📌 In MicroGPT (lines 116-119)
```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)   # Steps 1-2: mean of squares
    scale = (ms + 1e-5) ** -0.5               # Step 3: 1/sqrt(mean_square + epsilon)
    return [xi * scale for xi in x]            # Step 4: scale each element
```

---

## Step 12: Putting It All Together — The Full GPT

### 💬 Simple idea
The GPT model processes **one token at a time** and predicts what comes next. For each token:

```
Token comes in
  → Look up its embedding (what is this token?)
  → Add position embedding (where is it in the sequence?)
  → Normalize
  → Pass through attention (look at previous tokens)
  → Pass through MLP (process information)
  → Predict next token
```

### 🧮 The complete flow, step by step

```
Step 1: Token Embedding
   Input: token_id = 4 (the letter "e")
   Output: a vector of 16 numbers from the embedding table
   tok_emb = state_dict['wte'][4]   →  [0.12, -0.45, ..., 0.33]

Step 2: Position Embedding
   Input: pos_id = 0 (this is the first position)
   Output: a vector of 16 numbers
   pos_emb = state_dict['wpe'][0]   →  [0.05, 0.21, ..., -0.11]

Step 3: Add them together
   x[0] = tok_emb[0] + pos_emb[0] = 0.12 + 0.05 = 0.17
   x[1] = tok_emb[1] + pos_emb[1] = -0.45 + 0.21 = -0.24
   ... and so on for all 16 dimensions

Step 4: RMSNorm (keep numbers stable)

Step 5: Self-Attention (look at all previous tokens)
   5a. Compute Q, K, V using linear transformations
   5b. Store K and V for future tokens to use
   5c. Compute attention weights (Q · K, scale, softmax)
   5d. Weighted sum of V
   5e. Project output
   5f. Add residual connection (x = attention_output + x_before_attention)

Step 6: MLP (process the gathered information)
   6a. RMSNorm
   6b. Linear: 16 → 64 dimensions (expand)
   6c. ReLU (zero out negatives)
   6d. Linear: 64 → 16 dimensions (compress back)
   6e. Add residual connection

Step 7: Final prediction
   Linear: 16 → 27 dimensions (one score per possible character)
   These scores are called "logits"
```

### Residual connections — the "skip wire"

```
              ┌──────────────────────┐
              │                      │
  Input x ────┼──→ [Attention] ──→ (+) ──→ output
              │                      ↑
              └──────────────────────┘
                  "skip connection"

output = attention_result + original_x
```

Why? Without this, gradients get weaker as they pass through many layers (vanishing gradients). The skip connection gives the gradient a direct highway back to earlier layers.

### 📌 In MicroGPT (lines 121-156) — the gpt() function
```python
def gpt(token_id, pos_id, keys, values):
    # Steps 1-3: Embeddings
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # Step 5: Attention
        x_residual = x                    # save for skip connection
        x = rmsnorm(x)
        # ... attention computation (as described in Step 10) ...
        x = [a + b for a, b in zip(x, x_residual)]   # skip connection!

        # Step 6: MLP
        x_residual = x                    # save for skip connection
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 16 → 64
        x = [xi.relu() for xi in x]                       # ReLU
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 64 → 16
        x = [a + b for a, b in zip(x, x_residual)]       # skip connection!

    # Step 7: Final prediction
    logits = linear(x, state_dict['lm_head'])   # 16 → 27 logits
    return logits
```

---

# PART 3: Training and Generating Text

---

## Step 13: The Training Loop — Teaching the Model

### 💬 Simple idea
Training repeats these steps over and over:

```
For each training example (a name):
  1. Feed each character through the model one by one
  2. At each position, the model predicts the next character
  3. Measure how wrong the prediction was (loss)
  4. Use backpropagation to compute gradients
  5. Update the parameters to reduce the loss
  6. Reset gradients
  7. Print the loss so we can see progress
```

### 🧮 Tiny steps — one training step

Training on the name "emma":

```
Step 1: Tokenize
   "emma" → [BOS, 4, 12, 12, 0, BOS]
   (BOS = start/end marker, 4 = 'e', 12 = 'm', 0 = 'a')

Step 2: Create position pairs (input → target)
   Position 0:  input = BOS,  target = 'e'   (given start, predict 'e')
   Position 1:  input = 'e',  target = 'm'   (given 'e', predict 'm')
   Position 2:  input = 'm',  target = 'm'   (given 'm', predict 'm')
   Position 3:  input = 'm',  target = 'a'   (given 'm', predict 'a')
   Position 4:  input = 'a',  target = BOS   (given 'a', predict end)

Step 3: For each position, run the model and compute loss
   Position 0: model sees BOS, predicts prob distribution, loss₀ = -log(prob of 'e')
   Position 1: model sees 'e', predicts prob distribution, loss₁ = -log(prob of 'm')
   Position 2: model sees 'm', predicts prob distribution, loss₂ = -log(prob of 'm')
   Position 3: model sees 'm', predicts prob distribution, loss₃ = -log(prob of 'a')
   Position 4: model sees 'a', predicts prob distribution, loss₄ = -log(prob of BOS)

Step 4: Average all losses
   total_loss = (loss₀ + loss₁ + loss₂ + loss₃ + loss₄) / 5

Step 5: Backward pass — compute all gradients

Step 6: Update all parameters using Adam

Step 7: Reset all gradients to 0
```

### 📌 In MicroGPT (lines 153-184)
```python
for step in range(num_steps):     # repeat 1000 times

    # Step 1: Pick a name and tokenize it
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Step 3: Forward pass through each position
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id = tokens[pos_id]          # current token
        target_id = tokens[pos_id + 1]     # next token (the answer)
        logits = gpt(token_id, pos_id, keys, values)  # run the model
        probs = softmax(logits)                         # get probabilities
        loss_t = -probs[target_id].log()                # loss for this position
        losses.append(loss_t)

    # Step 4: Average loss
    loss = (1 / n) * sum(losses)

    # Step 5: Backward pass
    loss.backward()

    # Step 6: Update parameters (Adam optimizer)
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

        # Step 7: Reset gradient
        p.grad = 0

    print(f"step {step+1} / {num_steps} | loss {loss.data:.4f}")
```

---

## Step 14: Inference — Generating New Names

### 💬 Simple idea
After training, we can ask the model to generate new names it has never seen. The process:

```
Start with the BOS token
  → Model predicts probabilities for next character
  → Randomly pick a character (weighted by probabilities)
  → Feed that character back in
  → Repeat until the model outputs BOS (meaning "I'm done")
```

### 🧮 Tiny steps — generating one name

```
Step 1: Start with token = BOS
   Model predicts: {a: 0.15, b: 0.02, c: 0.01, ..., e: 0.30, ...}
   Randomly sample → picked 'e'

Step 2: Feed 'e' at position 1
   Model predicts: {a: 0.05, ..., m: 0.40, n: 0.10, ...}
   Randomly sample → picked 'm'

Step 3: Feed 'm' at position 2
   Model predicts: {a: 0.20, ..., m: 0.25, ...}
   Randomly sample → picked 'm'

Step 4: Feed 'm' at position 3
   Model predicts: {a: 0.45, ..., BOS: 0.05, ...}
   Randomly sample → picked 'a'

Step 5: Feed 'a' at position 4
   Model predicts: {BOS: 0.60, ...}
   Randomly sample → picked BOS (stop signal!)

Generated name: "emma"
```

### Temperature — controlling randomness

Before applying softmax, we divide the logits by a "temperature" number:

```
Temperature = 0.1 (very low):
   Logits:            [5.0, 3.0, 1.0]
   Divided by 0.1:    [50.0, 30.0, 10.0]   ← differences are AMPLIFIED
   After softmax:     [1.00, 0.00, 0.00]   ← almost always picks the top choice
   Result: very predictable, "safe" names

Temperature = 1.0 (normal):
   Logits:            [5.0, 3.0, 1.0]
   Divided by 1.0:    [5.0, 3.0, 1.0]      ← no change
   After softmax:     [0.84, 0.11, 0.02]   ← top choice likely but not guaranteed
   Result: diverse, sometimes surprising names

Temperature = 2.0 (high):
   Logits:            [5.0, 3.0, 1.0]
   Divided by 2.0:    [2.5, 1.5, 0.5]      ← differences are REDUCED
   After softmax:     [0.58, 0.24, 0.10]   ← more even distribution
   Result: very random, often weird names
```

### 📌 In MicroGPT (lines 189-200)
```python
temperature = 0.5

for sample_idx in range(20):      # generate 20 names
    # Fresh KV cache for each name
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = BOS          # start with BOS
    sample = []             # will collect characters

    for pos_id in range(block_size):    # up to 16 characters
        # Run the model
        logits = gpt(token_id, pos_id, keys, values)

        # Apply temperature: divide each logit by temperature
        scaled = []
        for l in logits:
            scaled.append(l / temperature)

        # Get probabilities
        probs = softmax(scaled)

        # Randomly pick the next token
        weights = []
        for p in probs:
            weights.append(p.data)
        token_id = random.choices(range(vocab_size), weights=weights)[0]

        # If model says "stop", stop!
        if token_id == BOS:
            break

        # Add the character to our generated name
        sample.append(uchars[token_id])

    print(f"sample {sample_idx+1}: {''.join(sample)}")
```

---

# PART 4: Quick Reference

---

## 🗺️ Where Every Line of MicroGPT Fits

| Lines | What It Does | Guide Step |
|---|---|---|
| 1-7 | Header and credits | — |
| 9-12 | Imports and random seed | Setup |
| 14-21 | Load and prepare dataset | Step 1 |
| 23-27 | Build tokenizer | Step 1 |
| 30-72 | **Autograd engine** | Step 9 |
| 74-90 | Initialize parameters | Step 3 (Vectors) |
| 94-106 | `linear` function | Step 5 (Matrix Multiply) |
| 110-114 | `softmax` function | Step 7 |
| 116-119 | `rmsnorm` function | Step 11 |
| 121-156 | **GPT forward pass** | Steps 10-12 |
| 158-161 | Adam optimizer setup | Step 9 |
| 163-196 | **Training loop** | Step 13 |
| 198-212 | **Inference** | Step 14 |

---

## 💡 How to Study This Guide

1. **Go in order.** Don't skip ahead. Each step builds on the previous one.
2. **Type every code example yourself.** Don't copy-paste. Typing builds understanding.
3. **Change the numbers.** After running an example, change a number and predict what will happen. Then run it.
4. **Add print statements.** When confused, add `print(variable_name)` after every line to see what's happening.
5. **Draw on paper.** For attention and backpropagation, draw the diagrams on paper with real numbers.
6. **Break things.** Remove RMSNorm. Delete residual connections. Set learning rate to 100. See what happens and WHY.
7. **Repeat.** Read each section at least twice. The second time will make much more sense.

---

## 📚 Resources (in learning order)

| Order | What | Why |
|---|---|---|
| 1 | [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Beautiful visual explanations, no code required |
| 2 | [Karpathy: micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0) | Builds the autograd engine from scratch |
| 3 | [Karpathy: makemore series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | Character-level language models, step by step |
| 4 | [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Visually explains every part of a transformer |
| 5 | [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Builds a fully working GPT from scratch |

---

*"You don't learn to ride a bicycle by reading about physics. You learn by falling off and getting back on. The same goes for code — run it, break it, fix it, repeat."*
