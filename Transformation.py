import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
print("Upload your TXT file containing 3D points:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
points = np.loadtxt(filename)
print("\nOriginal vectors:")
print(points)
print("\n========================================")
print("Enter the 3 x 3 transformation matrix")
print("========================================")
A = []
for i in range(3):
    row = list(map(float, input(f"Enter row {i+1}: ").split()))
    while len(row) != 3:
        print("Please enter exactly 3 numbers.")
        row = list(map(float, input(f"Enter row {i+1}: ").split()))
    A.append(row)
A = np.array(A)
print("\nTransformation Matrix A:")
print(A)
new_points = points @ A.T
print("\n========================================")
print("Transformed vectors")
print("========================================")
print(new_points)
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\n========================================")
print("Eigenvalues")
print("========================================")
print(eigenvalues)
print("\n========================================")
print("Eigenvectors")
print("========================================")
print(eigenvectors)
if np.max(np.abs(eigenvectors.imag)) > 1e-10:
    print("\nThe matrix has complex eigenvectors.")
    print("A real 3D eigenbasis cannot be plotted.")
else:
    P = eigenvectors.real
    determinant = np.linalg.det(P)
    if abs(determinant) < 1e-10:
        print("\nThe eigenvectors are not linearly independent.")
        print("Therefore, a complete eigenbasis does not exist.")
    else:
        eigen_points = np.linalg.solve(
            P,
            points.T
        ).T
        new_eigen_points = np.linalg.solve(
            P,
            new_points.T
        ).T
        print("\n========================================")
        print("Original vectors in Eigenbasis")
        print("========================================")
        print(eigen_points)
        print("\n========================================")
        print("Transformed vectors in Eigenbasis")
        print("========================================")
        print(new_eigen_points)
        print("\n========================================")
        print("Eigenvalue - Eigenvector pairs")
        print("========================================")
        for i in range(3):
            print("\nEigenvalue", i + 1, "=", eigenvalues[i])
            print("Eigenvector", i + 1, "=")
            print(P[:, i])
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(121, projection="3d")
        for i in range(len(points)):
            x = points[i, 0]
            y = points[i, 1]
            z = points[i, 2]
            ax1.quiver(
                0, 0, 0,
                x, y, z,
                arrow_length_ratio=0.08,
                linewidth=2,
                label="Original" if i == 0 else ""
            )
            x2 = new_points[i, 0]
            y2 = new_points[i, 1]
            z2 = new_points[i, 2]
            ax1.quiver(
                0, 0, 0,
                x2, y2, z2,
                arrow_length_ratio=0.08,
                linewidth=2,
                label="Transformed" if i == 0 else ""
            )
        ax1.set_title("Vectors in Normal Basis")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend()
        ax2 = fig.add_subplot(122, projection="3d")
        for i in range(len(eigen_points)):
            x = eigen_points[i, 0]
            y = eigen_points[i, 1]
            z = eigen_points[i, 2]
            ax2.quiver(
                0, 0, 0,
                x, y, z,
                arrow_length_ratio=0.08,
                linewidth=2,
                label="Original" if i == 0 else ""
            )
            x2 = new_eigen_points[i, 0]
            y2 = new_eigen_points[i, 1]
            z2 = new_eigen_points[i, 2]
            ax2.quiver(
                0, 0, 0,
                x2, y2, z2,
                arrow_length_ratio=0.08,
                linewidth=2,
                label="Transformed" if i == 0 else ""
            )
        ax2.set_title("Vectors in Eigenbasis")
        ax2.set_xlabel("Eigenvector 1")
        ax2.set_ylabel("Eigenvector 2")
        ax2.set_zlabel("Eigenvector 3")
        ax2.legend()
        plt.tight_layout()
        plt.show()