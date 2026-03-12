# ===== Simplified Implementation =====
def linear(x, w):

    output = []

    for row in w:  # each row of weights

        total = 0

        for wi, xi in zip(row, x):
            total += wi * xi

        output.append(total)

    return output


# Example usage:

# Input vector
x = [ 2, 3]

# Weight matrix (2 rows, 2 columns)
w = [
    [1, 2],
    [4, 5],
]

result = linear(x, w)
print(result)  # Output: [8,23]