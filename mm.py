# Python implementation of triple scalar product without using libraries

def cross_product(v1, v2):
    """Calculate the cross product of two 3D vectors"""
    return [
        v1[1] * v2[2] - v1[2] * v2[1],  # x component
        v1[2] * v2[0] - v1[0] * v2[2],  # y component
        v1[0] * v2[1] - v1[1] * v2[0]   # z component
    ]

def dot_product(v1, v2):
    """Calculate the dot product of two vectors"""
    return sum(a * b for a, b in zip(v1, v2))

def triple_scalar_product(a, b, c):
    """Calculate the triple scalar product A·(B×C)"""
    b_cross_c = cross_product(b, c)
    return dot_product(a, b_cross_c)

def determinant_3x3(matrix):
    """Calculate the determinant of a 3x3 matrix"""
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    
    return (a * (e * i - f * h) -
            b * (d * i - f * g) +
            c * (d * h - e * g))

# Define vectors
A = [1, 2, 3]
B = [4, 5, 6]
C = [7, 8, 9]

# Method 1: Using cross product and dot product
b_cross_c = cross_product(B, C)
result1 = dot_product(A, b_cross_c)

# Method 2: Using determinant directly
matrix = [A, B, C]
result2 = determinant_3x3(matrix)

# Print results
print("Vector A:", A)
print("Vector B:", B)
print("Vector C:", C)
print("Cross product B × C:", b_cross_c)
print("Triple scalar product A·(B×C) using cross and dot product:", result1)
print("Triple scalar product as determinant:", result2)