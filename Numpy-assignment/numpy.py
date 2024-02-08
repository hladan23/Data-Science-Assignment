import numpy as np 
# 2. Print the numpy version and the configuration
print("NumPy Version:", np.__version__)
print("NumPy Configuration:\n", np.show_config())

# 3. Create a null vector of size 10
null_vector = np.zeros(10)
print("Null Vector:", null_vector)

# 4. How to find the memory size of any array
array = np.array([1, 2, 3])
print("Memory size of the array:", array.nbytes, "bytes")

# 5. How to get the documentation of the numpy add function from the command line?
# (Use the following command in the command line)
# python -c "import numpy; numpy.info(numpy.add)"

# 6. Create a null vector of size 10 but the fifth value which is 1
null_vector_with_one = np.zeros(10)
null_vector_with_one[4] = 1
print("Null Vector with 1 at the fifth position:", null_vector_with_one)

# 7. Create a vector with values ranging from 10 to 49
vector_10_to_49 = np.arange(10, 50)
print("Vector from 10 to 49:", vector_10_to_49)

# 8. Reverse a vector (first element becomes last)
reversed_vector = np.flip(vector_10_to_49)
print("Reversed Vector:", reversed_vector)

# 9. Create a 3x3 matrix with values ranging from 0 to 8
matrix_3x3 = np.arange(9).reshape(3, 3)
print("3x3 Matrix:", matrix_3x3)

# 10. Find indices of non-zero elements from [1,2,0,0,4,0]
non_zero_indices = np.nonzero([1, 2, 0, 0, 4, 0])
print("Indices of non-zero elements:", non_zero_indices)

# 11. Create a 3x3 identity matrix
identity_matrix = np.identity(3)
print("3x3 Identity Matrix:\n", identity_matrix)

# 12. Create a 3x3x3 array with random values
random_3x3x3_array = np.random.random((3, 3, 3))
print("3x3x3 Array with Random Values:\n", random_3x3x3_array)

# 13. Create a 10x10 array with random values and find the minimum and maximum values
random_10x10_array = np.random.random((10, 10))
min_value = np.min(random_10x10_array)
max_value = np.max(random_10x10_array)
print("Minimum Value:", min_value)
print("Maximum Value:", max_value)

# 14. Create a 10x10 array with random values and find the minimum and maximum values
random_10x10_array = np.random.random((10, 10))
min_value = np.min(random_10x10_array)
max_value = np.max(random_10x10_array)
print("Minimum Value:", min_value)
print("Maximum Value:", max_value)

# 15. Create a 2d array with 1 on the border and 0 inside
border_array = np.ones((5, 5))
border_array[1:-1, 1:-1] = 0
print("2D Array with Border of 1 and Inside 0:\n", border_array)

# 16. Create a 2d array with 1 on the border and 0 inside
border_array = np.ones((5, 5))
border_array[1:-1, 1:-1] = 0
print("2D Array with Border of 1 and Inside 0:\n", border_array)

# 17. What is the result of the following expression?
# (printing each part)
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)

# 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
diagonal_below_matrix = np.diag(1 + np.arange(4), k=-1)
print("5x5 Matrix with Values 1,2,3,4 Below the Diagonal:\n", diagonal_below_matrix)

# 19. Create a 8x8 matrix and fill it with a checkerboard pattern
checkerboard_matrix = np.zeros((8, 8), dtype=int)
checkerboard_matrix[1::2, ::2] = 1
checkerboard_matrix[::2, 1::2] = 1
print("Checkerboard Pattern 8x8 Matrix:\n", checkerboard_matrix)

# 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
array_shape = (6, 7, 8)
index_100th_element = np.unravel_index(100, array_shape)
print("Index of the 100th Element:", index_100th_element)

# 21. Create a checkerboard 8x8 matrix using the tile function
checkerboard_tile = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print("Checkerboard Pattern 8x8 Matrix using Tile:\n", checkerboard_tile)
# 22. Normalize a 5x5 random matrix
random_5x5_matrix = np.random.random((5, 5))
normalized_matrix = (random_5x5_matrix - np.mean(random_5x5_matrix)) / np.std(random_5x5_matrix)
print("Normalized 5x5 Matrix:\n", normalized_matrix)
# 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA)
color_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')])
print("Custom Color dtype:", color_dtype)
# 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
matrix_5x3 = np.random.random((5, 3))
matrix_3x2 = np.random.random((3, 2))
product_matrix = np.dot(matrix_5x3, matrix_3x2)
print("Matrix Product of 5x3 and 3x2 matrices:\n", product_matrix)
# 25. Given a 1D array, negate all elements which are between 3 and 8, in place
array_1d = np.arange(10)
array_1d[(array_1d > 3) & (array_1d < 8)] *= -1
print("Negated elements between 3 and 8:", array_1d)
# 26. What is the output of the following script?
# Author: Jake VanderPlas
print(sum(range(5), -1))
from numpy import *
print(sum(range(5), -1))
# 27. Consider an integer vector Z, which of these expressions are legal?
# (Testing each expression individually)
Z = np.array([1, 2, 3])
print(Z**Z)
print(2 << Z >> 2)
print(Z < -Z)
print(1j * Z)
print(Z / 1 / 1)

# print(Z < Z > Z) (value error)
# 28. What are the result of the following expressions?
print(np.array(0) / np.array(0))     
print(np.array(0) // np.array(0))    
print(np.array([np.nan]).astype(int).astype(float))

# 29. How to round away from zero a float array?
float_array = np.random.uniform(-10, 10, 5)
rounded_array = np.copysign(np.ceil(np.abs(float_array)), float_array)
print("Original Float Array:", float_array)
print("Rounded Away from Zero Array:", rounded_array)

# 30. How to find common values between two arrays?
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([3, 4, 5, 6, 7])
common_values = np.intersect1d(array1, array2)
print("Common values between two arrays:", common_values)

# 31. How to ignore all numpy warnings (not recommended)?
# (Not recommended, but here is the code)
np.seterr(all='ignore')

# 32. Is the following expressions true?
print(np.sqrt(-1) == np.emath.sqrt(-1))

# 33. How to get the dates of yesterday, today and tomorrow?
yesterday = np.datetime64('today') - np.timedelta64(1, 'D')
today = np.datetime64('today')
tomorrow = np.datetime64('today') + np.timedelta64(1, 'D')
print("Yesterday:", yesterday)
print("Today:", today)
print("Tomorrow:", tomorrow)
# 34. How to get all the dates corresponding to the month of July 2016?
dates_july_2016 = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print("Dates in July 2016:", dates_july_2016)
# 35. How to compute ((A+B)*(-A/2)) in place (without copy)?
A = np.ones(3)
B = np.ones(3) * 2
np.multiply(np.add(A, B, out=B), np.negative(A, out=A), out=B)
print("Result of ((A+B)*(-A/2)):", B)
# 36. Extract the integer part of a random array of positive numbers using 4 different methods
random_positive_array = np.random.uniform(0, 10, 5)
integer_part_method1 = random_positive_array.astype(int)
integer_part_method2 = np.floor(random_positive_array)
integer_part_method3 = np.ceil(random_positive_array) - 1
integer_part_method4 = np.trunc(random_positive_array)
print("Original Random Positive Array:", random_positive_array)
print("Integer Part (Method 1):", integer_part_method1)
print("Integer Part (Method 2):", integer_part_method2)
print("Integer Part (Method 3):", integer_part_method3)
print("Integer Part (Method 4):", integer_part_method4)

# 37. Create a 5x5 matrix with row values ranging from 0 to 4
matrix_row_values = np.zeros((5, 5))
matrix_row_values += np.arange(5)
print("5x5 Matrix with Row Values 0 to 4:\n", matrix_row_values)
# 38. Consider a generator function that generates 10 integers and use it to build an array
generator_function = (x for x in range(10))
array_from_generator = np.fromiter(generator_function, int)
print("Array built from Generator Function:", array_from_generator)
# 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded
vector_excluded_values = np.linspace(0, 1, 12)[1:-1]
print("Vector with Values Ranging from 0 to 1 (Excluded):\n", vector_excluded_values)
# 40. Create a random vector of size 10 and sort it
random_vector_10 = np.random.random(10)
sorted_vector = np.sort(random_vector_10)
print("Random Vector of Size 10:", random_vector_10)
print("Sorted Vector:", sorted_vector)
# 41. How to sum a small array faster than np.sum?
small_array = np.arange(10)
sum_faster = np.add.reduce(small_array)
print("Sum of Small Array Faster than np.sum:", sum_faster)
# 42. Consider two random array A and B, check if they are equal
array_A = np.random.random(5)
array_B = np.random.random(5)
are_equal = np.array_equal(array_A, array_B)
print("Arrays A and B are equal:", are_equal)
# 43. Make an array immutable (read-only)
immutable_array = np.zeros(5)
immutable_array.flags.writeable = False
# Attempting to modify the array will raise an error
# immutable_array[0] = 1
# 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates
cartesian_coordinates = np.random.random((10, 2))
x, y = cartesian_coordinates[:, 0], cartesian_coordinates[:, 1]
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)
polar_coordinates = np.column_stack((r, theta))
print("Random Cartesian Coordinates:\n", cartesian_coordinates)
print("Converted Polar Coordinates:\n", polar_coordinates)
# 45. Create random vector of size 10 and replace the maximum value by 0
random_vector_replace_max = np.random.random(10)
random_vector_replace_max[random_vector_replace_max.argmax()] = 0
print("Random Vector with Maximum Value Replaced by 0:", random_vector_replace_max)
# 46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area
structured_array = np.zeros((5, 5), [('x', float), ('y', float)])
structured_array['x'], structured_array['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
print("Structured Array with x and y Coordinates:\n", structured_array)
# 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij = 1/(xi - yj))
X = np.arange(1, 6)
Y = X + 0.5
Cauchy_matrix = 1.0 / np.subtract.outer(X, Y)
print("Cauchy Matrix C:\n", Cauchy_matrix)
# 48. Print the minimum and maximum representable value for each numpy scalar type
for dtype in np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
    print("{dtype}: Min = {np.iinfo(dtype).min}, Max = {np.iinfo(dtype).max}")
# 49. How to print all the values of an array?
# (Setting print options to display all values)
np.set_printoptions(threshold=np.inf)
large_array = np.arange(100)
print("Large Array:", large_array)
# 50. How to find the closest value (to a given scalar) in a vector?
vector_closest_value = np.arange(10)
scalar_value = 7.5
closest_value = vector_closest_value[(np.abs(vector_closest_value - scalar_value)).argmin()]
print("Vector:", vector_closest_value)
print("Closest Value to", scalar_value, ":", closest_value)
# 51. Create a structured array representing a position (x,y) and a color (r,g,b)
position_color_array = np.zeros(5, dtype=[('position', [('x', float), ('y', float)]), ('color', [('r', int), ('g', int), ('b', int)])])
print("Structured Array with Position and Color:\n", position_color_array)
# 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances
coordinates_vector = np.random.random((100, 2))
distances = np.linalg.norm(coordinates_vector[:, np.newaxis] - coordinates_vector, axis=-1)
print("Random Coordinates Vector:\n", coordinates_vector)
print("Distances between Points:\n", distances)
# 53. How to convert a float (32 bits) array into an integer (32 bits) in place?
float_array_32 = np.random.uniform(0, 10, 5).astype(np.float32)
integer_array_32 = float_array_32.view(np.int32)
print("Float (32 bits) Array:\n", float_array_32)
print("Converted Integer (32 bits) Array:\n", integer_array_32)
# 54. How to read the following file? (★★☆)
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
# ,  , 9,10,11
# 55. What is the equivalent of enumerate for numpy arrays?
array_for_enumerate = np.array(['apple', 'banana', 'cherry'])
for index, value in np.ndenumerate(array_for_enumerate):
    print(f"Index: {index}, Value: {value}")

# 56. Generate a generic 2D Gaussian-like array
x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
d = np.sqrt(x*x + y*y)
sigma, mu = 1.0, 0.0
gaussian_array = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
print("2D Gaussian-like Array:\n", gaussian_array)

# 57. How to randomly place p elements in a 2D array?
p_elements = 3
array_2d_random_p_elements = np.zeros((5, 5))
indices_to_fill = np.random.choice(np.arange(5*5), p_elements, replace=False)
array_2d_random_p_elements.flat[indices_to_fill] = 1
print(f"2D Array with {p_elements} Randomly Placed Elements:\n", array_2d_random_p_elements)

# 58. Subtract the mean of each row of a matrix
matrix_to_subtract_mean = np.random.random((3, 3))
mean_per_row = matrix_to_subtract_mean.mean(axis=1)
matrix_subtracted_mean = matrix_to_subtract_mean - mean_per_row[:, np.newaxis]
print("Original Matrix:\n", matrix_to_subtract_mean)
print("Matrix with Mean of Each Row Subtracted:\n", matrix_subtracted_mean)
# 59. How to sort an array by the nth column?
array_to_sort_by_column = np.array([[1, 3, 2], [5, 1, 6], [2, 7, 4]])
n_column_to_sort_by = 1
sorted_array_by_column = array_to_sort_by_column[array_to_sort_by_column[:, n_column_to_sort_by].argsort()]
print(f"Array Sorted by {n_column_to_sort_by}th Column:\n", sorted_array_by_column)

# 60. How to tell if a given 2D array has null columns?
array_with_null_columns = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
null_columns = ~array_with_null_columns.any(axis=0)
print("Array with Null Columns:\n", array_with_null_columns)
print("Null Columns:", null_columns)

# 61. Find the nearest value from a given value in an array
array_nearest_value = np.array([1, 2, 3, 4, 5])
given_value = 3.7
nearest_value = array_nearest_value[(np.abs(array_nearest_value - given_value)).argmin()]
print("Array:", array_nearest_value)
print(f"Nearest Value to {given_value}:", nearest_value)

# 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator?
array_1 = np.arange(3).reshape(1, 3)
array_2 = np.arange(3).reshape(3, 1)
iterator_sum = np.nditer([array_1, array_2, None])
for a, b, result in iterator_sum:
    result[...] = a + b
print("Array 1:\n", array_1)
print("Array 2:\n", array_2)
print("Sum using Iterator:\n", iterator_sum.operands[2])

# 63. Create an array class that has a name attribute
class NamedArray(np.ndarray):
    def __new__(cls, array, name=""):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
# Example usage:
named_array = NamedArray([1, 2, 3], name="ExampleArray")
print("Named Array:", named_array)
print("Name attribute:", named_array.name)

# 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)?
given_vector = np.array([1, 2, 3, 4, 5])
indices_to_add_one = np.array([1, 3, 1, 4])
given_vector += np.bincount(indices_to_add_one, minlength=len(given_vector))
print("Given Vector with 1 added to specified indices:", given_vector)

# 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?
vector_X = np.array([1, 2, 3, 4, 5])
index_list_I = np.array([1, 3, 1, 4])
accumulated_array_F = np.bincount(index_list_I, weights=vector_X)
print("Vector X:", vector_X)
print("Index List I:", index_list_I)
print("Accumulated Array F:", accumulated_array_F)

# 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors
image_shape = (5, 5, 3)
image_dtype = np.uint8
image_array = np.random.randint(0, 256, size=image_shape, dtype=image_dtype)
unique_colors = len(np.unique(image_array.reshape(-1, 3), axis=0))
print("Number of Unique Colors in Image:", unique_colors)

# 67. Considering a four dimensions array, how to get sum over the last two axis at once?
four_dimensions_array = np.random.random((2, 3, 4, 5))
sum_over_last_two_axes = four_dimensions_array.sum(axis=(-2, -1))
print("Original 4D Array Shape:", four_dimensions_array.shape)
print("Sum over Last Two Axes:", sum_over_last_two_axes)

# 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices?
vector_D = np.array([1, 2, 3, 4, 5, 6])
vector_S = np.array([0, 0, 1, 1, 2, 2])
means_of_subsets = np.bincount(vector_S, weights=vector_D) / np.bincount(vector_S)
print("Vector D:", vector_D)
print("Vector S:", vector_S)
print("Means of Subsets:", means_of_subsets)
# 69. How to get the diagonal of a dot product?
matrix_A = np.random.random((3, 3))
matrix_B = np.random.random((3, 3))
dot_product_diagonal = np.diag(np.dot(matrix_A, matrix_B))
print("Matrix A:\n", matrix_A)
print("Matrix B:\n", matrix_B)
print("Diagonal of Dot Product:", dot_product_diagonal)

# 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value?
original_vector = np.array([1, 2, 3, 4, 5])
zeros_interleaved_vector = np.zeros(len(original_vector) + (len(original_vector) - 1) * 3)
zeros_interleaved_vector[::4] = original_vector
print("Original Vector:", original_vector)
print("New Vector with Zeros Interleaved:", zeros_interleaved_vector)
# 71. Consider an array of dimension (5,5,3), how to multiply it by an array with dimensions (5,5)?
array_3d = np.random.random((5, 5, 3))
array_2d = np.random.random((5, 5))
result_multiplication = array_3d * array_2d[:, :, np.newaxis]
print("3D Array Shape:", array_3d.shape)
print("2D Array Shape:", array_2d.shape)
print("Result of Multiplication:", result_multiplication)

# 72. How to swap two rows of an array?
array_to_swap_rows = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array_to_swap_rows[[0, 1]] = array_to_swap_rows[[1, 0]]
print("Array with Rows Swapped:\n", array_to_swap_rows)
# 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices),
# find the set of unique line segments composing all the triangles
triplets_set = set(tuple(map(tuple, np.random.randint(1, 10, (10, 3)))))
lines_set = set(tuple(sorted([a, b])) for triplet in triplets_set for a, b in zip(triplet, triplet[1:] + triplet[:1]))
print("Set of Unique Line Segments from Triangles:\n", lines_set)

# 74. Given a sorted array C that corresponds to a bincount, how to produce an array A
# such that np.bincount(A) == C?
sorted_bincount_array = np.array([1, 2, 0, 0, 4, 0])
result_array_A = np.repeat(np.arange(len(sorted_bincount_array)), sorted_bincount_array)
print("Sorted Bincount Array:\n", sorted_bincount_array)
print("Result Array A:\n", result_array_A)

# 75. How to compute averages using a sliding window over an array?
array_for_sliding_window = np.arange(10)
window_size = 3
averages_with_sliding_window = np.convolve(array_for_sliding_window, np.ones(window_size)/window_size, mode='valid')
print("Original Array:\n", array_for_sliding_window)
print("Averages with Sliding Window (size = {}):\n".format(window_size), averages_with_sliding_window)

# 76. Consider a one-dimensional array Z, build a two-dimensional array whose
# first row is (Z[0], Z[1], Z[2]) and each subsequent row is shifted by 1
array_for_shifted_rows = np.arange(1, 11)
shifted_rows_array = np.lib.stride_tricks.sliding_window_view(array_for_shifted_rows, (3,))
print("Original Array for Shifted Rows:\n", array_for_shifted_rows)
print("Resultant 2D Array with Shifted Rows:\n", shifted_rows_array)

# 77. How to negate a boolean, or to change the sign of a float inplace?
boolean_array_to_negate = np.array([True, False, True])
negated_boolean_array = np.logical_not(boolean_array_to_negate)
print("Original Boolean Array:\n", boolean_array_to_negate)
print("Negated Boolean Array:\n", negated_boolean_array)
# 78. Consider 2 sets of points P0, P1 describing lines (2d) and a point p,
# how to compute distance from p to each line i (P0[i], P1[i])?
P0 = np.array([[0, 0], [1, 1], [2, 2]])
P1 = np.array([[1, 0], [0, 1], [-1, 0]])
p = np.array([1, 1])
distances_to_lines = np.abs(np.cross(P1 - P0, P0 - p)) / np.linalg.norm(P1 - P0, axis=1)
print("Set of Points P0:\n", P0)
print("Set of Points P1:\n", P1)
print("Point P:\n", p)
print("Distances from P to Lines (P0[i], P1[i]):\n", distances_to_lines)

# 79. Consider 2 sets of points P0, P1 describing lines (2d) and a set of points P,
# how to compute distance from each point j (P[j]) to each line i (P0[i], P1[i])?
P = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
distances_from_points_to_lines = np.array([np.abs(np.cross(P1 - P0, P0 - p)) / np.linalg.norm(P1 - P0, axis=1)
                                           for p in P])
print("Set of Points P:\n", P)
print("Distances from Points to Lines (P0[i], P1[i]):\n", distances_from_points_to_lines)

# 80. Consider an arbitrary array, write a function that extracts a subpart with a fixed shape
# and centered on a given element (pad with a fill value when necessary)
def extract_subpart(array, shape, center):
    indices = np.arange(center - shape // 2, center + shape // 2)
    indices = np.clip(indices, 0, len(array))
    subpart = np.full(shape, fill_value=array[0, 0])
    subpart[indices] = array[indices]
    return subpart

array_for_subpart_extraction = np.random.randint(1, 10, (7, 7))
shape_of_subpart = (3, 3)
center_of_subpart = (3, 3)
subpart_result = extract_subpart(array_for_subpart_extraction, shape_of_subpart, center_of_subpart)
print("Original Array for Subpart Extraction:\n", array_for_subpart_extraction)
print("Resultant Subpart with Shape {} Centered at {}:\n".format(shape_of_subpart, center_of_subpart), subpart_result)
# 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array
# R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]?
original_array_Z = np.arange(1, 15)
array_R = np.lib.stride_tricks.sliding_window_view(original_array_Z, (4,))
print("Original Array Z:\n", original_array_Z)
print("Resultant Array R:\n", array_R)

# 82. Compute a matrix rank
matrix_for_rank = np.random.random((4, 4))
rank_of_matrix = np.linalg.matrix_rank(matrix_for_rank)
print("Matrix for Rank Computation:\n", matrix_for_rank)
print("Rank of Matrix:", rank_of_matrix)

# 83. How to find the most frequent value in an array?
array_for_most_frequent_value = np.array([1, 2, 3, 2, 2, 1, 3, 4, 2, 4, 2, 5])
most_frequent_value = np.bincount(array_for_most_frequent_value).argmax()
print("Array for Most Frequent Value:\n", array_for_most_frequent_value)
print("Most Frequent Value:", most_frequent_value)

# 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix
random_10x10_matrix = np.random.random((10, 10))
contiguous_blocks_3x3 = np.lib.stride_tricks.sliding_window_view(random_10x10_matrix, (3, 3))
print("Random 10x10 Matrix:\n", random_10x10_matrix)
print("Contiguous 3x3 Blocks:\n", contiguous_blocks_3x3)

# 85. Create a 2D array subclass such that Z[i,j] == Z[j,i]
class SymmetricArray(np.ndarray):
    def __setitem__(self, index, value):
        super().__setitem__((index[1], index[0]), value)
symmetric_array = SymmetricArray((3, 3), dtype=int)
symmetric_array[0, 1] = 1
symmetric_array[1, 2] = 2
symmetric_array[2, 0] = 3
print("Symmetric Array (Z[i,j] == Z[j,i]):\n", symmetric_array)

# 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1).
# How to compute the sum of the p matrix products at once? (result has shape (n,1))
p_matrices = [np.random.random((3, 3)) for _ in range(5)]
p_vectors = [np.random.random((3, 1)) for _ in range(5)]
sum_of_matrix_products = np.sum([np.dot(matrix, vector) for matrix, vector in zip(p_matrices, p_vectors)], axis=0)
print("Set of Matrices (p) with Shape (3, 3):\n", p_matrices)
print("Set of Vectors (p) with Shape (3, 1):\n", p_vectors)
print("Sum of Matrix Products (Result Shape: (3, 1)):\n", sum_of_matrix_products)

# 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)?
array_16x16 = np.random.random((16, 16))
block_sum_4x4 = np.lib.stride_tricks.sliding_window_view(array_16x16, (4, 4)).sum(axis=(2, 3))
print("Original 16x16 Array:\n", array_16x16)
print("Block-Sum with Block Size 4x4:\n", block_sum_4x4)

# 88. How to implement the Game of Life using numpy arrays?
def game_of_life(board, generations):
    for _ in range(generations):
        neighbors_count = sum(np.roll(np.roll(board, i, 0), j, 1)
                             for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))
        board = (neighbors_count == 3) | (board & (neighbors_count == 2))
    return board

initial_board = np.random.choice([0, 1], size=(10, 10))
generations_to_simulate = 5
final_board = game_of_life(initial_board, generations_to_simulate)
print("Initial Game of Life Board:\n", initial_board)
print("Final Game of Life Board after {} Generations:\n".format(generations_to_simulate), final_board)
# 89. How to get the n largest values of an array?
array_for_n_largest_values = np.random.random(10)
n_largest_values_indices = np.argpartition(array_for_n_largest_values, -3)[-3:]
n_largest_values = array_for_n_largest_values[n_largest_values_indices]
print("Array for N Largest Values:\n", array_for_n_largest_values)
print("Indices of N Largest Values:", n_largest_values_indices)
print("N Largest Values:", n_largest_values)

# 90. Given an arbitrary number of vectors, build the cartesian product
# (every combination of every item)
vectors_for_cartesian_product = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
cartesian_product = np.array(np.meshgrid(*vectors_for_cartesian_product)).T.reshape(-1, len(vectors_for_cartesian_product))
print("Vectors for Cartesian Product:\n", vectors_for_cartesian_product)
print("Cartesian Product:\n", cartesian_product)

# 91. How to create a record array from a regular array?
regular_array_for_record_array = np.array([(1, 'John', 25), (2, 'Jane', 30), (3, 'Doe', 22)],
                                          dtype=[('ID', int), ('Name', 'U10'), ('Age', int)])
record_array = np.rec.array(regular_array_for_record_array)
print("Regular Array for Record Array:\n", regular_array_for_record_array)
print("Record Array:\n", record_array)

# 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods
large_vector_Z = np.random.random(1000000)
method1_result = np.power(Z, 3)
method2_result = Z ** 3
method3_result = np.einsum('i,i,i->i', Z, Z, Z)
print("Large Vector Z (First 5 Elements):\n", large_vector_Z[:5])
print("Result of Z to the Power of 3 (Method 1):\n", method1_result[:5])
print("Result of Z to the Power of 3 (Method 2):\n", method2_result[:5])
print("Result of Z to the Power of 3 (Method 3):\n", method3_result[:5])
# 89. How to get the n largest values of an array?
array_for_n_largest_values = np.random.random(10)
n_largest_values_indices = np.argpartition(array_for_n_largest_values, -3)[-3:]
n_largest_values = array_for_n_largest_values[n_largest_values_indices]
print("Array for N Largest Values:\n", array_for_n_largest_values)
print("Indices of N Largest Values:", n_largest_values_indices)
print("N Largest Values:", n_largest_values)

# 90. Given an arbitrary number of vectors, build the cartesian product
# (every combination of every item)
vectors_for_cartesian_product = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
cartesian_product = np.array(np.meshgrid(*vectors_for_cartesian_product)).T.reshape(-1, len(vectors_for_cartesian_product))
print("Vectors for Cartesian Product:\n", vectors_for_cartesian_product)
print("Cartesian Product:\n", cartesian_product)

# 91. How to create a record array from a regular array?
regular_array_for_record_array = np.array([(1, 'John', 25), (2, 'Jane', 30), (3, 'Doe', 22)],
                                          dtype=[('ID', int), ('Name', 'U10'), ('Age', int)])
record_array = np.rec.array(regular_array_for_record_array)
print("Regular Array for Record Array:\n", regular_array_for_record_array)
print("Record Array:\n", record_array)

# 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods
large_vector_Z = np.random.random(1000000)
method1_result = np.power(Z, 3)
method2_result = Z ** 3
method3_result = np.einsum('i,i,i->i', Z, Z, Z)
print("Large Vector Z (First 5 Elements):\n", large_vector_Z[:5])
print("Result of Z to the Power of 3 (Method 1):\n", method1_result[:5])
print("Result of Z to the Power of 3 (Method 2):\n", method2_result[:5])
print("Result of Z to the Power of 3 (Method 3):\n", method3_result[:5])
# 93. Consider two arrays A and B of shape (8,3) and (2,2).
# How to find rows of A that contain elements of each row of B regardless of the order of the elements in B?
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
              [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]])
B = np.array([[2, 1], [8, 7]])
rows_containing_elements_of_B = np.all(np.isin(A, B), axis=1)
print("Array A (Shape: {}):\n".format(A.shape), A)
print("Array B (Shape: {}):\n".format(B.shape), B)
print("Rows of A Containing Elements of B:\n", rows_containing_elements_of_B)

# 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3])
matrix_for_unequal_rows = np.array([[1, 1, 1], [2, 2, 3], [4, 5, 6], [7, 7, 7],
                                   [8, 8, 9], [10, 11, 12], [13, 13, 13], [14, 15, 16], [17, 18, 19], [20, 21, 21]])
unequal_rows = np.any(np.diff(matrix_for_unequal_rows, axis=1) != 0, axis=1)
print("Original Matrix for Unequal Rows (10x3):\n", matrix_for_unequal_rows)
print("Rows with Unequal Values:\n", unequal_rows)

# 95. Convert a vector of ints into a matrix binary representation
vector_for_binary_representation = np.array([1, 2, 3, 4], dtype=np.uint8)
binary_representation_matrix = np.unpackbits(vector_for_binary_representation[:, np.newaxis], axis=1)
print("Vector for Binary Representation:\n", vector_for_binary_representation)
print("Matrix Binary Representation:\n", binary_representation_matrix)

# 96. Given a two-dimensional array, how to extract unique rows?
two_dimensional_array_for_unique_rows = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9], [4, 5, 6]])
unique_rows = np.unique(two_dimensional_array_for_unique_rows, axis=0)
print("Original 2D Array for Unique Rows:\n", two_dimensional_array_for_unique_rows)
print("Unique Rows:\n", unique_rows)
# 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function
A_for_einsum = np.array([1, 2, 3])
B_for_einsum = np.array([4, 5, 6])
inner_product = np.einsum('i,i', A_for_einsum, B_for_einsum)
outer_product = np.einsum('i,j->ij', A_for_einsum, B_for_einsum)
sum_result = np.einsum('i', A_for_einsum)
elementwise_product = np.einsum('i,i->i', A_for_einsum, B_for_einsum)
print("Vector A for Einsum:\n", A_for_einsum)
print("Vector B for Einsum:\n", B_for_einsum)
print("Einsum Equivalent of Inner Product:\n", inner_product)
print("Einsum Equivalent of Outer Product:\n", outer_product)
print("Einsum Equivalent of Sum:\n", sum_result)
print("Einsum Equivalent of Elementwise Product:\n", elementwise_product)

# 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples?
X_for_path_sampling = np.array([1, 2, 3, 4, 5])
Y_for_path_sampling = np.array([10, 15, 7, 12, 8])
equidistant_samples = np.linspace(0, 1, 100)
sampled_path = np.column_stack([np.interp(equidistant_samples, np.linspace(0, 1, len(X_for_path_sampling)), X_for_path_sampling),
                               np.interp(equidistant_samples, np.linspace(0, 1, len(Y_for_path_sampling)), Y_for_path_sampling)])
print("Vector X for Path Sampling:\n", X_for_path_sampling)
print("Vector Y for Path Sampling:\n", Y_for_path_sampling)
print("Equidistant Samples:\n", equidistant_samples)
print("Sampled Path:\n", sampled_path)
# 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as
# draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers
# and which sum to n.
n_for_multinomial = 5
array_X_for_multinomial = np.array([[1, 2, 2], [2, 1, 2], [0, 5, 0], [3, 1, 1], [1, 1, 3], [0, 2, 3]])
valid_rows_for_multinomial = (array_X_for_multinomial.sum(axis=1) == n_for_multinomial) & np.all(array_X_for_multinomial.astype(int) == array_X_for_multinomial, axis=1)
selected_rows_for_multinomial = array_X_for_multinomial[valid_rows_for_multinomial]
print("Integer n for Multinomial Distribution:", n_for_multinomial)
print("2D Array X for Multinomial Distribution:\n", array_X_for_multinomial)
print("Rows that can be interpreted as draws from Multinomial Distribution:\n", selected_rows_for_multinomial)

# 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X
array_X_for_bootstrapping = np.random.normal(0, 1, 1000)
bootstrap_samples = np.random.choice(array_X_for_bootstrapping, (10000, len(array_X_for_bootstrapping)), replace=True)
confidence_interval_lower, confidence_interval_upper = np.percentile(bootstrap_samples.mean(axis=1), [2.5, 97.5])
print("1D Array X for Bootstrapping (First 5 Elements):\n", array_X_for_bootstrapping[:5])
print("Bootstrapped 95% Confidence Interval for the Mean:\n", (confidence_interval_lower, confidence_interval_upper))

















    
    
    
    
    
    
    
    

























    
    



























































































































