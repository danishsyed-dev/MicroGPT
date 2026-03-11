vector_a = [4, 5]
vector_b = [2, 3]

# Step 1: Multiply matching positions and add them up
score_part_1 = vector_a[0] * vector_b[0]  # 4 * 2 = 8
score_part_2 = vector_a[1] * vector_b[1]  # 5 * 3 = 15

# Step 2: Final sum
final_score = score_part_1 + score_part_2 # 8 + 15 = 23
print(f"Final score: {final_score}")