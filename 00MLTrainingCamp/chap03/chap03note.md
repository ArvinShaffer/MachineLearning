[TOC]

# 广播算法

$$
X = \begin{bmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \\ x_{31} & x_{32} \end{bmatrix} \\
\quad \\
\beta = \begin{bmatrix}b_{11} & b_{12} \\ b_{21} & b_{22} \end{bmatrix}
$$

求和形式可以转换成向量相乘
$$
Z_{ij} = \sum_{k}x_{ik}\beta_{kj} = x_i^t \beta_j
$$

$$
\sum_k x_k \beta_k = x^t \beta
$$

$$
X = \begin{bmatrix} x_1^t \\ x_2^t \\ x_3^t \end{bmatrix} \\
\quad \\
\beta = \begin{bmatrix} \beta_1 & \beta_2\end{bmatrix}
$$

```
x[: None] -> 10 X 1
y -> 4
```

# Einsum

- 1.$x$ is  $I \times J \times K$, $y$  is  $J \times K \times H$ and $z$   is  $I \times J \times H$.
- 2.To work out the expression, we have $z_{ijh} = ? x_{ij} \cdot y_{jh}$.
- 3.Now since $k$ is missing from the r.h.s.,  we must  have it in the sum. In other words $z_{ijh} = \sum_{k = 1}^{K} x_{ijk} y_{jkh}$.

```
I, J, K, H = 10, 15, 20, 25
x = np.random.randn(I, J, K)
y = np.random.randn(I, J, H)
z = np.einsum('ijk, jkh -> ijh', x, y)
print(z)
```
- 1.$x: I \times J \times K$,   $y: J \times K \times H$,  $z: I \times J \times H$

- 2.$z:einsum("ijk, ikh -> ijh", x, y)$

- 3.$z_{ijh} = \sum_k x_{ijk}y_{ikh}$


linformer    capsule network

https://zhuanlan.zhihu.com/p/101157166

```
I, J, K, H = 10, 15, 20, 25
x = np.random.randn(I, J, K)
y = np.random.randn(I, J, H)
z = np.einsum('ijk, jkh -> ih', x, y)
print(z)
```

- 1.$x: I \times J \times K$,   $y: J \times K \times H$,  $z: I \times H$

- 2.$z_{ih} = ?x_{i..}y_{..h}$

- 3.$z_{ih} = \sum_j \sum_k x_{ijk}y_{jkh}$

求对角线

```
einsum('ii -> i'...)
```



```
einsum('i,j -> ij', x, y)
```

1.$x:I, y:J, z:I \times J$

2.$z_{ij} = \sum x_i y_j$

```
x:64 X 10 X 5
y:5 X 10
z = x@y
```

广播配合einsum求z
$$
X: 64 \times 10 \times 5    \qquad B \times I \times J   \\
Y: 64 \times 5 \times 10    \qquad B \times J \times K   \\
Z: 64 \times 10 \times 10   \qquad B \times I \times K   \\
$$

- 1.$\sum_j x_{bij}y_{jk}$
- 2.einsum('bij, jk->bik', x, y)



根据einsum反推广播形式
$$
I, J, K, H = 10, 15, 20, 23 \\
X: I \times J \times K   \\
Y: J \times K \times H \\
Z: einsum('ijk, jkh -> ijh', x, y)
$$
首先把einsum写成求和的形式，然后再进行求解
$$
z_{ijh} = \sum_k x_{ijk}y_{jkh} \\
        = x_{ij}^Ty_{jh} \\
\tilde{x} = IJ \times K, \tilde{y} = K \times JH \\
IJ \times JH ? \rightarrow x_{ij}y_{j'h} \quad \\
\quad \\
x:ijk \rightarrow jik ;\quad y:jkh \\
x@y\cdot jik \rightarrow ijh
$$


















































