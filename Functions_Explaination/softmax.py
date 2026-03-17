import math  # Used for the exponential function(e^x), where value of e is approx 2.71828

"""
Softmax

Plain English explanation: 
Softmax is a mathematical function that converts any random list of numbers into clean probabilities (percentages) that sum up to exactly 1 (which is the same as 100%).

Simple real-world analogy: 
Imagine three friends bidding on the last slice of pizza using arbitrary point values: Alice bids 2 points, Bob bids 1 point, and Charlie bids 0.5 points. Softmax looks at those random points and converts them into the exact percentage chance each friend has of winning the pizza, ensuring the total chance equals exactly 100%.

Numeric example with small numbers (expanded step-by-step): 
Let's use the input [2.0, 1.0, 0.5].

Step 1: The model applies the exponential function to each number (this makes the big numbers stand out even more).
Exponential of 2.0 ≈ 7.39
Exponential of 1.0 ≈ 2.72
Exponential of 0.5 ≈ 1.65

Step 2: It adds those results together to get a total.
7.39 + 2.72 + 1.65 = 11.76

Step 3: It divides each individual exponential by the total sum so they all become fractions of 1.
7.39 / 11.76 = 0.63 (or 63%)
2.72 / 11.76 = 0.23 (or 23%)
1.65 / 11.76 = 0.14 (or 14%)
Our final vector is [0.63, 0.23, 0.14], which beautifully sums to 1.0!
"""

# Simple Python code:

points = [2.0, 1.0, 0.5]

# Step 1: Calculate exponentials
exps = [math.exp(points[0]), math.exp(points[1]), math.exp(points[2])] # e^2.0, e^1.0, e^0.5

# Step 2: Calculate the total sum
total = sum(exps) # e^2.0 + e^1.0 + e^0.5

# Step 3: Divide each by the total
probabilities = [exps[0]/total, exps[1]/total, exps[2]/total] # e^2.0 / total, e^1.0 / total, e^0.5 / total

print(f"Points: {points}")
print(f"Probabilities: {probabilities}")


"""
MicroGPT code reference: In microgpt.py, lines 97-101 handle this:

def softmax(logits): 
    max_val = max(val.data for val in logits) 
    exps = [(val - max_val).exp() for val in logits] 
    total = sum(exps) 
    return [e / total for e in exps]

Note: You'll see max_val subtracted here. This is just a safety trick to keep the computer from dealing with math numbers that are too massively large to fit in memory, but the mathematical percentages at the end come out exactly the same!
"""
