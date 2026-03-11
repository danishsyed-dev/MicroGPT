vector_a = [4, 5]
vector_b = [2, 3]

# Step 1: Multiply matching positions and add them up
score_part_1 = vector_a * vector_b  # 2 * 4 = 8
score_part_2 = vector_a[2] * vector_b[2]  # 3 * 1 = 3

# Step 2: Final sum
final_score = score_part_1 + score_part_2 # 8 + 3 = 11