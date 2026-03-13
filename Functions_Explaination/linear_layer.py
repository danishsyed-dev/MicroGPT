
# ===== Original Implementation (Line 138) =====
#linear(x, state_dict[f'layer{li}.mlp_fc1'])

# ===== Simplified Implementation (Expanded) =====
input_vector = [2, 3]
weights_matrix = [
    [4, 5], 
    [6, 7]
]

# The linear layer simply performs matrix multiplication
output_vector = [
    (4 * 2) + (5 * 3), # 23
    (6 * 2) + (7 * 3)  # 33
]
print(output_vector) # [23, 33]