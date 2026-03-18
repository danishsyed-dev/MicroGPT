"""
The most atomic way to train and inference a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

# Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()] # list[str] of documents
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to discrete symbols and back
# Review Question 1: How does MicroGPT convert a letter like 'e' into a number?
# Plain English explanation: Before doing any math, the model looks through all the text it has been given and pulls out every unique character. It sorts these characters (usually alphabetically) and assigns a simple counting number to each one, starting from 0. So, if 'e' happens to be the 5th letter in that sorted list, its specific number (token ID) becomes 4.
# Simple real-world analogy: Think of a teacher creating an alphabetical class roster. Every student gets a roll number based on their alphabetical order. MicroGPT does the exact same thing for letters.
# Step 1: set(''.join(docs)) merges all the text together and strips out the duplicates so we only have unique characters left.
# Step 2: sorted(...) puts them in order.
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
# Step 3: The model creates one extra number to represent the BOS (Beginning of Sequence), and sets the total vocab_size to be the number of characters plus one.
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be Autograd, to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    # Checkpoint Question 2: Where is ReLU used in the model, why is it needed, and what would happen if we removed it?
    # Plain English explanation: ReLU is our "bouncer" rule: keep positive numbers the same, but turn negative numbers into zeros. If we removed it, the model would become painfully simple and entirely lose its ability to understand complex text.
    # Why it's needed (The What-If): If you stack multiple linear layers on top of each other without a rule like ReLU in between them, mathematically, they just collapse into the equivalent of one single, giant linear layer. It would be like trying to build a complex machine out of straight pipes without using any elbow joints; you could never bend the pipes to create complex shapes. ReLU breaks up the straight lines, giving the model the ability to learn highly complex, non-linear patterns.
    # The Rule (Line 50): max(0, self.data). This simple line uses Python's built-in max function to say "pick whichever is higher: zero, or the number." If the number is negative, zero is higher!
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model.
# Review Question 2: What does n_embd = 16 mean and why is it important?
# Plain English explanation: A single ID number isn't enough to capture how a letter behaves in different words. n_embd = 16 tells the model that every single token must be represented by a detailed list of exactly 16 numbers. This is important because it gives the AI 16 different "dials" or "traits" to describe and relate that letter to other letters.
# Simple real-world analogy: Imagine grading a movie. Giving it a single 5-star rating (like a token ID) doesn't tell you much. But grading it across 16 different categories like acting, music, writing, and pacing (an embedding vector) gives you a highly detailed description.
# This sets the rule that all vectors will be 16 numbers long.
n_embd = 16     # embedding dimension
n_head = 4      # number of attention heads
n_layer = 1     # number of layers
block_size = 16 # maximum sequence length
head_dim = n_embd // n_head # dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# Define the model architecture: a stateless function mapping token sequence and parameters to logits over what comes next.
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
# Checkpoint Question 3: Show me the linear function in microgpt.py and walk through it step by step with a numeric example.
# Plain English explanation: The linear function takes an input vector (a single list of numbers) and transforms it into a brand new vector. It does this by using a matrix (a stack of vectors, often called "weights") and calculating the dot product between your input vector and every single row in that matrix.
# Simple real-world analogy: If a dot product is comparing your food preferences to one friend to get a similarity score, the linear function is comparing your food preferences to a whole classroom of friends. The output is a brand new list containing your similarity score for each person in the room.
# Numeric example with small numbers: Imagine your input vector x is [1, 2]. Imagine your weight matrix w has two rows:
# Row 1: [3, 4]
# Row 2: [5, 6]
# Step 1: Dot product of x and Row 1 -> (1 * 3) + (2 * 4) = 3 + 8 = 11
# Step 2: Dot product of x and Row 2 -> (1 * 5) + (2 * 6) = 5 + 12 = 17
# Final Output: Your new vector is [11, 17].
def linear(x, w): 
    # Let's break this into tiny steps:
    # for wo in w: The model looks at the matrix w and loops through it one row (wo) at a time.
    # zip(wo, x): For the current row, it takes a number from the row and pairs it with the matching number in your input vector x.
    # wi * xi: It multiplies those paired numbers together.
    # sum(...): It adds all those multiplied numbers up to get a single score for that row (completing the dot product).
    # [...]: Finally, it wraps all those individual scores into a brand new list.
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

# Checkpoint Question 1: Walk through the softmax function in microgpt.py with the numbers [3.0, 1.0, 0.5]
# Plain English explanation: As we learned, softmax takes a list of numbers and turns them into a list of percentages (probabilities) that add up to exactly 1.0 (or 100%).
def softmax(logits):
    # Step-by-step with real numbers: Let's trace exactly what the computer does with our input [3.0, 1.0, 0.5]:
    
    # Step 1 (max_val = ...): The model finds the highest number to use as a safety anchor so the math doesn't get too huge. The highest number is 3.0.
    max_val = max(val.data for val in logits)
    
    # Step 2 (val - max_val): It subtracts 3.0 from every number. Our list becomes [0.0, -2.0, -2.5].
    # Step 3 (.exp()): It applies the exponential math function to each of these new numbers.
    # Exponential of 0.0 = 1.0
    # Exponential of -2.0 ≈ 0.135
    # Exponential of -2.5 ≈ 0.082
    exps = [(val - max_val).exp() for val in logits]
    
    # Step 4 (total = sum(exps)): It adds these together. 1.0 + 0.135 + 0.082 =  1.217.
    total = sum(exps)
    
    # Step 5 (e / total): It divides each individual exponential by the total sum to get the final percentages.
    # 1.0 / 1.217 ≈ 0.82 (82%)
    # 0.135 / 1.217 ≈ 0.11 (11%)
    # 0.082 / 1.217 ≈ 0.07 (7%)
    # Our final output is [0.82, 0.11, 0.07], which beautifully sums up to 1.0!
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    # Later, the model actually uses this when a letter enters the network:
    tok_emb = state_dict['wte'][token_id] # token embedding {Here, state_dict['wte'] acts like a massive lookup table. The model plugs in the single token_id and pulls out the specific 16-number vector (tok_emb) assigned to that letter}
    
    pos_emb = state_dict['wpe'][pos_id] # position embedding {Here, state_dict['wpe'] acts like a massive lookup table. The model plugs in the single pos_id and pulls out the specific 16-number vector (pos_emb) assigned to that position}

    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding {Here, we add the token embedding and position embedding to get the final embedding (x)}
    
    x = rmsnorm(x) # {Here, we apply RMSNorm to the embedding (x) to normalize it}

    for li in range(n_layer):
        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            # Checkpoint Question 4: What is a dot product? Show me where it appears in the attention code.
            # Plain English explanation: As we've seen, the dot product takes two vectors of the same size and returns a single number that measures how similar they are.
            # Simple real-world analogy: Think of a matching game where you line up two different blocks of colors. For every color that perfectly matches in the exact same spot, you get points. You add up all those points to get your final "match score."
            # The calculation: sum(q_h[j] * k_h[t][j] for j in range(head_dim))
            # Let's break down exactly what this is doing:
            # Step 1: The model has two vectors it wants to compare: q_h (a "Query" representing what the current token is looking for) and k_h[t] (a "Key" representing what a past token contains).
            # Step 2: for j in range(head_dim) loops through every position in these vectors one by one.
            # Step 3: q_h[j] * k_h[t][j] multiplies the numbers at the matching positions together.
            # Step 4: sum(...) adds all those smaller multiplications into one final score. This score tells the model how strongly the current token should "attend" to or focus on that past token.
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        # The Application (Line 139): [xi.relu() for xi in x]. This loops through every single number (xi) in an entire list (x) and applies that bouncer rule to it.
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters.
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients.
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")