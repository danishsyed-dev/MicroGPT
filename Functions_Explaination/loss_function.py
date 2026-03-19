import math

# The model is only 20% sure that 'e' is the correct next letter
probability_of_correct_answer = 0.2 

# Calculate the penalty (loss) using the negative logarithm
loss = -math.log(probability_of_correct_answer)  # The model is penalized for being wrong

print(loss)
