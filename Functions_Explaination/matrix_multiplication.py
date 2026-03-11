my_vector = [4, 5]
friend_matrix = [
    [2, 3], # Friend 1
    [1, 2]  # Friend 2
]

# Do a dot product for each row
score_1 = (my_vector[0] * friend_matrix[0][0]) + (my_vector[1] * friend_matrix[0][1]) # 4*2 + 5*3 = 23
score_2 = (my_vector[0] * friend_matrix[1][0]) + (my_vector[1] * friend_matrix[1][1]) # 4*1 + 5*2 = 14

final_vector = [score_1, score_2] # [23, 14]
print(f"Final vector: {final_vector}")