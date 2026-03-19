# ===== Original MicroGPT Implementation (Line 94-95) =====
# def linear(x, w):
#     return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

# ===== Simplified Implementation (Expanded) =====
x = [3,4] # input vector


# weights matrix
w = [
    [5,6], # weights for first output
    [1,2]  # weights for second output
]


# Linear function
def linear(x, w):

    output = []                     # This will store our final result

    for wo in w:                    # 1. Loop through each row of weights

        total = 0                   # 2. Initialize a running total

        for wi, xi in zip(wo, x):   # 3. Pair up (weight, input)
            total += wi * xi        # 4. multiply + add them up (weights * inputs)

        output.append(total)        # 5. Save the result

    return output

print(linear(x, w))