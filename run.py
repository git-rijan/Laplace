from classes import Boundary,Generate_Pandas,Triangulate,Calculation,Solution

x_coords = [0, 1, 2,1.5]
y_coords = [0, 1, 0,-2]
values = [10, 20, 30,40]

num_of_points_inside=100

# Example usage
boundary = Boundary(x_coords, y_coords, values)
boundary.plot()

generator_pandas = Generate_Pandas(boundary, num_of_points_inside)

triangulate = Triangulate(boundary, generator_pandas)
triangulate.plot()

# calculation = Calculation(triangulate)

# solution=Solution(calculation,boundary,triangulate)
# print(solution.calculate_solution())
# solution.surface()
# solution.color()
