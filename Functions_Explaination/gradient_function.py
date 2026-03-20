"""
Gradients

Plain English explanation: A gradient is simply an indicator that tells the model which direction to adjust a number (a parameter) to make the loss smaller. It answers the question: "If I nudge this specific parameter up a tiny bit, does my penalty go up or down?"

If the gradient is positive, increasing the parameter increases the loss (which is bad!), so the model knows it needs to decrease the parameter.

If the gradient is negative, increasing the parameter decreases the loss (which is good!), so the model knows it needs to increase the parameter.

Simple real-world analogy: Think of a gradient like using a compass when you are lost in the mountains trying to reach the valley bottom (which represents zero loss). The gradient tells you if taking a step forward goes uphill (bad, positive gradient) or downhill (good, negative gradient).

Numeric example with small numbers: Let's say a specific weight in our model is currently 0.5, and its gradient is calculated to be 2.0. Because the gradient is a positive number, it means increasing the weight pushes the loss up. So, the model knows it must step in the opposite direction, adjusting the weight down to something like 0.4.

MicroGPT code reference: In MicroGPT, gradients are so incredibly important that every single number in the network gets a dedicated variable just to track its gradient. This is set up right at the beginning of the file in the Value class, around line 27: 
class Value: 
    __slots__ = ('data', 'grad', '_children', '_local_grads')

- data holds the actual weight number (like 0.5).
- grad holds the gradient number telling it which way to adjust (like 2.0).
"""

# Example 1: The "Hill" (Gradient is positive)
weight = 0.5
gradient = 2.0

if gradient > 0:
    # The slope is going uphill, so we must step backwards (decrease)!
    weight = weight - 0.1 
elif gradient < 0:
    # The slope is going downhill, so we continue forward (increase)!
    weight = weight + 0.1

print(f"Final Weight: {weight}")    

# Example 2: The "Mountain Peak" (Gradient is 0)
weight = 0.5
gradient = 0.0

# If the gradient is 0, we are at the bottom of the valley (or the top of a hill).
# We stop moving because we can't go any lower/higher.
weight = weight - 0.1 # (or + 0.1, it doesn't matter)

print(f"Final Weight: {weight}")
