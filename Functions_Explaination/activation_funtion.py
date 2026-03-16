"""
If our input vector is [-3, 5, -1, 8], ReLU transforms it into [0, 5, 0, 8].
Simple Python code:
"""

input_numbers = [-3, 5, -1, 8]
output_numbers = []

for number in input_numbers:
    if number > 0:
        output_numbers.append(number) # Keep positive numbers
    else:
        output_numbers.append(0)      # Turn negative numbers to 0

print(f"Input: {input_numbers}")
print(f"Output: {output_numbers}")

"""
MicroGPT code reference: You can find ReLU defined around line 50 of the code: max(0, self.data)
This is a clever Python trick. It compares 0 to the data number and picks whichever is higher. If the number is -5, 0 is higher, so it becomes 0.
Later, around line 139, the model applies this rule to a whole vector: [xi.relu() for xi in x]
This loops through every single number (xi) in the vector x and applies the bouncer rule.
"""
