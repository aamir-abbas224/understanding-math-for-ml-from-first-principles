# Linear Algebra

## Pre-Requisites

- Python installed  
- NumPy (v2.2.6 used in examples)

> **Note:** Examples can be found in the `sub-folder`.

---

## What is Linear Algebra?

**Linear Algebra** is the study of vectors and the rules to manipulate them.  
Vectors are denoted as *→v* or *v*. Numerical data is represented as **vectors**, and a table of such data is represented as a **matrix**.  

*Why it matters:* Machine Learning data is often stored as vectors or matrices. Operations like dot products, matrix multiplication, eigenvalues, and transformations are used everywhere (e.g., linear regression, PCA, neural networks).

---

## Systems of Linear Equations

A **system of linear equations** is a collection of two or more linear equations involving the same unknown variables, considered simultaneously. Each equation represents a linear relationship, and the goal is to find values of the variables that satisfy all equations at once.

Mathematically:

\[
\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{aligned}
\]

- \(x_1, \dots, x_n\) — unknown variables  
- \(a_{ij}\) — coefficients  
- \(b_i\) — constants  

### Matrix Form

\[
\mathbf{A}\mathbf{x} = \mathbf{b}
\]

- **A**: coefficient matrix  
- **x**: vector of unknowns  
- **b**: outcome or constraint vector  

*Geometric Interpretation:* Each linear equation defines a line or plane. The solution set is the intersection of these lines/planes (a point, line, or empty set).

---

## Matrices

A **matrix** is a rectangular arrangement of numbers, symbols, or expressions organized into rows and columns.

\[
\mathbf{A} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
\]

- \(a_{ij}\): element in the \(i\)-th row and \(j\)-th column  
- \(m\): number of rows  
- \(n\): number of columns  

**Interpretation:**  

- Collection of vectors (rows or columns)  
- Linear transformation mapping input vectors to output vectors  
- Compact representation of a system of linear equations  

---

## Matrix Operations

### Matrix Addition

For matrices of the same shape:

\[
A + B =
\begin{bmatrix}
a_{11}+b_{11} & \cdots & a_{1n}+b_{1n} \\
\vdots & \ddots & \vdots \\
a_{m1}+b_{m1} & \cdots & a_{mn}+b_{mn}
\end{bmatrix}
\]

### Matrix Multiplication

For \(A_{m \times n}\) and \(B_{n \times p}\):

\[
C = AB
\]

> Note: Matrix multiplication is **not element-wise**. Element-wise multiplication is the **Hadamard product**.

---

## NumPy

**NumPy** provides multi-dimensional arrays and operations for numerical computing:

- Vectors, matrices, tensors in any dimension  
- Element-wise operations: +, -, *, /  
- Linear algebra utilities (`np.linalg`) — inverse, determinant, eigenvalues, solving systems  
- Efficient handling of large datasets  
- Foundation for ML libraries (TensorFlow, PyTorch, pandas)

---

## Identity Matrix

An \(n \times n\) matrix with 1s on the diagonal and 0 elsewhere.

---

## Matrix Properties

### Associativity

\[
\forall A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p}, C \in \mathbb{R}^{p \times q} : (AB)C = A(BC)
\]

### Distributivity

\[
(A + B)C = AC + BC
\]

\[
A(C + D) = AC + AD
\]

### Multiplication with Identity

\[
I_m A = A I_n = A \quad (I_m \neq I_n \text{ if } m \neq n)
\]

---

## Inverse Matrix

- Only square matrices can have an inverse  
- Exists if determinant ≠ 0 (matrix is **invertible**)  
- Denoted \(A^{-1}\), used in linear regression  

\[
AA^{-1} = I = A^{-1}A
\]

\[
(AB)^{-1} = B^{-1}A^{-1}, \quad (\text{not } A^{-1}+B^{-1})
\]

---

## Transpose

\(\mathbf{A}^T\) — obtained by swapping rows and columns  

Properties:

\[
(\mathbf{A}^T)^T = \mathbf{A}, \quad (\mathbf{A}\mathbf{B})^T = \mathbf{B}^T\mathbf{A}^T, \quad (\mathbf{A}+\mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T
\]

**Symmetric Matrix:** Square matrix equal to its transpose  

- Covariance and Hessian matrices are symmetric  
- Real eigenvalues and orthogonal eigenvectors  

---

## Row-Echelon Form

- Non-zero rows above zero rows  
- Leading entry (pivot) of each row to the right of the one above  
- Entries below pivots = 0  

**Reduced Row-Echelon Form:**  

- Each pivot = 1  
- Pivot is the only non-zero in its column  

---

## Gaussian Elimination

Method to solve linear systems by transforming a matrix to row-echelon form.

---

## Augmented Matrix

Combines coefficients and constants into one matrix for row operations.

---

## Pivot

Leading coefficient of a row (first non-zero number from left), forming a **staircase structure** in row-echelon form.

---

## Vector Space

A set \(V\) with operations **vector addition** and **scalar multiplication** is a vector space over field \(F\) if it satisfies:

1. Closure under addition  
2. Closure under scalar multiplication  
3. Addition rules: associativity, commutativity, zero vector, additive inverses  
4. Scalar rules: distributivity and compatibility with field multiplication  

---

## Linear Combination

A linear combination of vectors is formed by multiplying each vector by a scalar and adding the results:

\[
y = \lambda_1 v_1 + \lambda_2 v_2
\]

- \(v_1, v_2 \in V\)  
- \(\lambda_1, \lambda_2 \in \mathbb{R}\)
