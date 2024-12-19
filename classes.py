import numpy as np
import scipy.linalg
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point, Polygon
import triangle as tr
import geopandas as gpd
import pandas as pd
import time


# Boundary class for polygon
class Boundary:
    def __init__(self, x, y, value):
        if len(x) != len(y) or len(y) != len(value):
            raise ValueError("All input arrays (x, y, value) must have the same length.")
        self.x = np.array(x)
        self.y = np.array(y)
        self.value = np.array(value)

    def plot(self):
        plt.figure(figsize=(6, 6))
        plt.plot(
            np.append(self.x, self.x[0]),
            np.append(self.y, self.y[0]),
            marker='o',
            linestyle='-',
            linewidth=2.5,
            markersize=8
        )
        plt.fill(np.append(self.x, self.x[0]), np.append(self.y, self.y[0]), alpha=0.2)

        # Label boundary points without 'u=value'
        for i, (xi, yi) in enumerate(zip(self.x, self.y), start=1):
            plt.text(xi, yi, f'(x{i},y{i})', fontsize=12, fontweight='bold', ha='right', va='bottom')

        plt.axis('off')
        plt.axis('equal')
        plt.show()

    def __repr__(self):
        return f"Boundary(x={self.x}, y={self.y}, value={self.value})"

# Generate points inside based on Pandas/GeoPandas logic
class Generate_Pandas:
    def __init__(self, boundary_obj, num_of_points_inside):
        self.boundary = boundary_obj
        self.polygon = Polygon(zip(self.boundary.x, self.boundary.y))
        self.num_of_points_inside = num_of_points_inside
        
        # Generate points when the object is initialized
        self._generate_points()
    
    def _generate_points(self):
        x, y = self.Random_Points_in_Bounds(self.polygon, self.num_of_points_inside)
        
        # Create a DataFrame and filter points inside the polygon
        df = pd.DataFrame({'points': list(zip(x, y))})
        df['points'] = df['points'].apply(Point)
        gdf_points = gpd.GeoDataFrame(df, geometry='points')

        # Create GeoDataFrame for the polygon
        gdf_poly = gpd.GeoDataFrame(index=["myPoly"], geometry=[self.polygon])

        # Perform spatial join to find points within the polygon
        Sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, predicate="within", how='left')

        # Keep points within the polygon
        pnts_in_poly = gdf_points[Sjoin.index_right == 'myPoly']

        # Store the x and y coordinates of the points inside the polygon
        self.x = pnts_in_poly.geometry.x.values
        self.y = pnts_in_poly.geometry.y.values

    def Random_Points_in_Bounds(self, polygon, number):
        """
        Generate random points within the bounding box of the polygon.
        """
        minx, miny, maxx, maxy = polygon.bounds
        x = np.random.uniform(minx, maxx, number)
        y = np.random.uniform(miny, maxy, number)
        return x, y
    
    def plot(self):
        # Plot the boundary
        plt.figure(figsize=(6, 6))
        plt.plot(
            np.append(self.boundary.x, self.boundary.x[0]),
            np.append(self.boundary.y, self.boundary.y[0]),
            marker='o',
            linestyle='-',
            linewidth=2.5,
            markersize=8
        )
        plt.fill(np.append(self.boundary.x, self.boundary.x[0]), np.append(self.boundary.y, self.boundary.y[0]), alpha=0.2)

        # Plot the scatter of points inside the polygon
        plt.scatter(self.x, self.y, color='red', label='Generated Points', zorder=5)

        # Label boundary points without 'u=value'
        for i, (xi, yi) in enumerate(zip(self.boundary.x, self.boundary.y), start=1):
            plt.text(xi, yi, f'(x{i},y{i})', fontsize=12, fontweight='bold', ha='right', va='bottom')

        plt.axis('off')
        plt.axis('equal')
        plt.legend(loc='upper left')
        plt.show()

    def timeit(self):
        """
        Calculate the time taken to generate the points inside the polygon.
        """
        start_time = time.time()  # Start time
        self._generate_points()   # Call the point generation method
        end_time = time.time()    # End time
        time_taken = end_time - start_time  # Time taken to run _generate_points
        
        return time_taken

#Generate meshes using triangle library
class Triangulate:
    def __init__(self, boundary, generator_pandas):
        """
        Initialize the Triangulate class with boundary and generator objects.

        :param boundary: Object containing boundary x and y coordinates.
        :param generator_pandas: Object containing generator x and y coordinates.
        """
        self.boundary = boundary
        self.generator_pandas = generator_pandas
        self.all_indices = None  # Attribute to store triangle indices
        self.all_triangles = None  # Attribute to store triangle vertex coordinates
        self.boundary_only_indices = None  # Indices for boundary points
        self.inner_only_indices = None  # Indices for inner generator points
        self.total_points = 0  # Attribute to store the total number of points
        self.triangulate()

    def vertices(self):
        """
        Combine x and y coordinates from boundary and generator to create vertices as a NumPy array.
        
        :return: 2D NumPy array of vertices where each row is [x, y].
        """
        x_combined = np.concatenate([self.boundary.x, self.generator_pandas.x])
        y_combined = np.concatenate([self.boundary.y, self.generator_pandas.y])
        
        # Stack x and y as columns to create a 2D array
        vertices_array = np.column_stack((x_combined, y_combined))
        
        # Update total_points attribute
        self.total_points = len(vertices_array)
        
        return vertices_array

    @staticmethod
    def bndry_segment(n):
        """
        Generate boundary segments for n points using NumPy operations.
        
        :param n: Number of points.
        :return: Array of boundary segments as tuples.
        """
        if n < 2:
            raise ValueError("n must be at least 2")

        # Create an array of sequential indices
        indices = np.arange(n)
        # Create an array of start and end indices (including the wrap-around)
        segments = np.column_stack((indices, np.roll(indices, -1)))
        
        return segments

    def edges(self):
        """
        Generate edges using the boundary object.

        :return: Array of edges.
        """
        n = len(self.boundary.x)  # Assuming the number of boundary points is determined by x-coordinates
        return self.bndry_segment(n)

    def triangulate(self):
        """
        Perform the triangulation process and store triangle indices as an attribute.

        :return: Dictionary containing triangulation data.
        """
        vertices = self.vertices()
        edges = self.edges()
        n_boundary = len(self.boundary.x)  # Number of boundary points
        n_combined = len(vertices)  # Total number of points

        # Set boundary and inner indices
        self.boundary_only_indices = np.arange(n_boundary)
        self.inner_only_indices = np.arange(n_boundary, n_combined)

        # Prepare input data for triangulation
        input_data = {'vertices': vertices, 'segments': edges}
        triangulation = tr.triangulate(input_data, 'p')

        # Store triangle indices and triangles
        self.all_indices = triangulation['triangles']
        self.all_triangles = triangulation['vertices'][triangulation['triangles']]

        return triangulation

    def plot(self):
        """
        Plot the triangulation using tr.compare and Matplotlib.
        """
        vertices = self.vertices()
        edges = self.edges()
        input_data = {'vertices': vertices, 'segments': edges}

        # Perform triangulation
        triangulation = self.triangulate()

        # Use tr.compare for visualization
        tr.compare(plt, input_data, triangulation)
        plt.show()

#FEM Calculation
class Calculation:
    def __init__(self, triangulate):
        """
        Initialize the Solution class with a Triangulate object.

        :param triangulate: An instance of the Triangulate class.
        """
        self.triangulate = triangulate

    @staticmethod
    def Le_func(arr, n):
        """
        Create an Le matrix for a given array of indices.

        :param arr: Array of vertex indices for a triangle.
        :param n: Total number of points.
        :return: A 3xN NumPy matrix for the triangle.
        """
        matrix = np.zeros((3, n))
        matrix[0, arr[0]] = 1
        matrix[1, arr[1]] = 1
        matrix[2, arr[2]] = 1
        return matrix

    @staticmethod
    def area_func(arr):
        """
        Calculate the area of a triangle given the vertex coordinates.

        :param arr: Array of vertex coordinates for a triangle.
        :return: The area of the triangle.
        """
        return 0.5 * np.abs(
            arr[0, 0] * (arr[1, 1] - arr[2, 1]) +
            arr[1, 0] * (arr[2, 1] - arr[0, 1]) +
            arr[2, 0] * (arr[0, 1] - arr[1, 1])
        )

    @staticmethod
    def Be_func(arr):
        """
        Create a Be matrix for a given array of vertex coordinates for a triangle.

        :param arr: Array of vertex coordinates for a triangle.
        :return: A 2x3 NumPy matrix for the triangle.
        """
        Be_matrix = np.array([
            [arr[1, 1] - arr[2, 1], arr[2, 1] - arr[0, 1], arr[0, 1] - arr[1, 1]],
            [arr[2, 0] - arr[1, 0], arr[0, 0] - arr[2, 0], arr[1, 0] - arr[0, 0]]
        ])
        return Be_matrix

    @staticmethod
    def ke_func(Be_matrix, area):
        """
        Compute the Ke matrix for a given Be matrix and area.

        :param Be_matrix: The Be matrix for a triangle.
        :param area: The area of the triangle.
        :return: The Ke matrix.
        """
        return 0.25 * np.dot(Be_matrix.T, Be_matrix) / area

    def Le_matrices(self):
        """
        Compute Le matrices for all triangles in the triangulate object.

        :return: A NumPy array of Le matrices.
        """
        if self.triangulate.all_indices is None:
            raise ValueError("Triangulation must be performed before computing Le matrices.")
        
        n_points = self.triangulate.total_points
        triangle_indices = self.triangulate.all_indices

        # Compute Le matrices for all triangles
        self.Le_matrices_values = np.array([self.Le_func(arr, n_points) for arr in triangle_indices])
        return self.Le_matrices_values

    def area_all(self):
        """
        Compute areas for all triangles in the triangulate object.

        :return: A NumPy array of areas for all triangles.
        """
        if self.triangulate.all_triangles is None:
            raise ValueError("Triangulation must be performed before calculating areas.")
        
        all_triangles = self.triangulate.all_triangles

        # Calculate areas for all triangles using area_func
        self.area_all_values = np.array([self.area_func(arr) for arr in all_triangles])
        return self.area_all_values

    def Be_matrices(self):
        """
        Compute Be matrices for all triangles in the triangulate object.

        :return: A NumPy array of Be matrices.
        """
        if self.triangulate.all_triangles is None:
            raise ValueError("Triangulation must be performed before calculating Be matrices.")
        
        all_triangles = self.triangulate.all_triangles

        # Calculate Be matrices for all triangles using Be_func
        self.Be_matrices_array = np.array([self.Be_func(arr) for arr in all_triangles])
        return self.Be_matrices_array

    def Ke_matrices(self):
        """
        Compute Ke matrices for all triangles in the triangulate object.

        :return: A NumPy array of Ke matrices.
        """
        if self.triangulate.all_triangles is None:
            raise ValueError("Triangulation must be performed before calculating Ke matrices.")
        
        all_triangles = self.triangulate.all_triangles
        # Get the Be matrices
        Be_all = self.Be_matrices()
        # Get the areas of all triangles
        area_all = self.area_all()

        # Calculate Ke matrices for all triangles using ke_func
        self.Ke_matrices_values = np.array([self.ke_func(Be, area) for Be, area in zip(Be_all, area_all)])
        return self.Ke_matrices_values

    def K_all_global(self):
        """
        Compute the global stiffness matrix K_all_global by accumulating the result of matrix multiplications.

        :return: A NumPy array representing the global stiffness matrix.
        """
        # Ensure necessary matrices are computed
        Le_all = self.Le_matrices()  # Accessing Le_matrices_values now
        ke_all = self.Ke_matrices()  # Accessing Ke_matrices_values now

        # Transpose Le_all for efficient computation
        Le_all_trans = np.transpose(Le_all, (0, 2, 1))
        
        # Initialize the global stiffness matrix
        K_all_global = np.zeros((Le_all.shape[2], Le_all.shape[2]))

        # Perform matrix multiplication slice by slice and accumulate in K_all_global
        for i in range(ke_all.shape[0]):
            # Compute the temporary matrix product
            temp = np.matmul(np.matmul(Le_all_trans[i], ke_all[i]), Le_all[i])

            # Add the result to the global stiffness matrix
            K_all_global += temp

        return K_all_global

#Solution Calculation
class Solution:
    def __init__(self, calculation, boundary, triangulate):
        # Store the calculation, boundary, and triangulate objects
        self.calculation = calculation
        self.boundary = boundary
        self.triangulate = triangulate

    def calculate_solution(self):
        # Get the global matrix from the calculation object
        K_all_global = self.calculation.K_all_global()

        # Access the required properties from the boundary and triangulate objects
        inner_only_indices = self.triangulate.inner_only_indices
        boundary_only_indices = self.triangulate.boundary_only_indices
        value = self.boundary.value

        # Slice the global matrix K_all_global using the provided indices
        sliced_matrix_1 = K_all_global.take(inner_only_indices, axis=0).take(inner_only_indices, axis=1)
        sliced_matrix_2 = K_all_global.take(inner_only_indices, axis=0).take(boundary_only_indices, axis=1)

        # Compute the right-hand side vector B
        B = -1 * np.dot(sliced_matrix_2, value)
        
        # Solve the linear system
        sol = scipy.linalg.solve(sliced_matrix_1, B)
        
        # Get the vertices from triangulate object
        vertices = self.triangulate.vertices()

        # Combine boundary values and solution
        sol_dict = {tuple(pair_2d): pair_1d for pair_2d, pair_1d in zip(vertices, np.concatenate((value, sol)))}

        return sol_dict

    def color(self):
        # Get the solution dictionary
        sol_dict = self.calculate_solution()

        # Extract the x, y coordinates and corresponding solution values
        x_array = [key[0] for key in sol_dict.keys()]
        y_array = [key[1] for key in sol_dict.keys()]
        value_array = list(sol_dict.values())

        # Determine the min and max values for the color map
        min_value = np.min(value_array)
        max_value = np.max(value_array)
        step = 0.1

        # Generate color map with evenly spaced colors
        cmap = plt.get_cmap('viridis', int((max_value - min_value) / step) + 1)

        # Define the grid for interpolation
        xi = np.linspace(min(x_array), max(x_array), 500)
        yi = np.linspace(min(y_array), max(y_array), 500)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate the solution values onto the grid
        zi = griddata((x_array, y_array), value_array, (xi, yi), method='cubic')

        # Create the interpolated color plot
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(xi, yi, zi, levels=100, cmap=cmap)
        cbar = plt.colorbar(contour)
        cbar.set_label('Solution Value')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Interpolated Color Plot of Solution')
        plt.show()

    def surface(self):
        # Get the solution dictionary
        sol_dict = self.calculate_solution()

        # Extract the x, y coordinates and corresponding solution values
        x_array = [key[0] for key in sol_dict.keys()]
        y_array = [key[1] for key in sol_dict.keys()]
        value_array = list(sol_dict.values())

        # Determine the min and max values for the surface plot
        min_value = np.min(value_array)
        max_value = np.max(value_array)

        # Define the grid for interpolation
        xi = np.linspace(min(x_array), max(x_array), 500)
        yi = np.linspace(min(y_array), max(y_array), 500)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate the solution values onto the grid
        zi = griddata((x_array, y_array), value_array, (xi, yi), method='cubic')

        # Create the interpolated 3D surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Solution Value')
        # ax.set_title('Interpolated 3D Surface Plot')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()














