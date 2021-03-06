[TOC]

# 逻辑回归的实现

**从模型到代码的过程**

- 非常精细的写出模型当中的每一步
- 检查是否有标记的错误
- 使用推导当中的写法，忽略 pep8 进行开发
- 使用最笨的方式进行开发，不要考虑效率
- 使用 Monte Carlo 检查简单的模型是否正常

**复习**

- 极大似然的概念
-  矩阵求导的基本法则

**逻辑回归基本设定**
- 定义$\sigma : x \mapsto \frac{1}{1 + exp(-x)}$
- 逻辑回归的概率密度函数为 $p_{\beta} (x_i ) = \sigma(x^t_i \beta)$， 其中 $\beta$为未知参数，$x_i$为解释变量
- 负的对数似然函数为 $−\sum_i y_i \log(p_{\beta} (x_i )) + (1 − y_i ) log(1 − p_{\beta} (x_i ))$
- 我们现在需要做的是求他的导数

**最麻烦的部分**

- 由于矩阵形式非常简单，所以最麻烦的部分其实是对一堆非线性的函数的推导
- 我们当然可以手推，但是问题在于，手推很容易出错
- 所以这时候 sympy 就可以出场了
- 见 notebook

**使用 SymPy 之后**
我们可以写出对数似然函数的导数为
$$
-\sum_i (y_i exp(-x_i^t \beta)/(1 + exp(-x_i^t \beta)) - (1 - y_i)exp(x_i^t \beta)/(1 + exp(x_i^t \beta)))x_i
$$
括号里面的东西还是有点复杂，所以我们不妨再试试看 sympy 是否能帮我们化简

**化简结果如下**
$$
- \sum_i (y_i - \sigma(x_i^t \beta))x_i
$$
使用 jax 实现自动求导的过程并测试整体的正确性（见 colab notebook）。

## Jax

- 请注意在这里，我们尽可能用接近于 Numpy 的形式进行了实现，并且我们通过 Jax 的 Autograd 机制判断了我们是否求导准确，这一点是非常重要的
- 如果没有 Jax，我们可能只能用 PyTorch 或者 TF 计算 Autograd（会相对来说更麻烦一些）
- 如果这些 Autograd 的机制都没有这时候我们面临的问题就更为麻烦，这时候通常使用 Finite Difference 进行调试
- 请注意这种调试往往会造成很大的误差，所以有时候很难进行判断
- 在我们上一讲当中，我们解释过如果我们得到了导数，接下来我们该使用一些优化方法来一步步进行优化了
- 不幸的是，就目前对于 Python 来说，不论是 jax.scipy.opimitize.minimize还是 scipy.optimize.minimize 都有巨大的问题
- 就 Jax 来说，大部分优化算法都没有实现，实现的部分也有 bug
- 就 scipy 来说，问题更大一些

**Scipy 的问题**

- Scipy 优化的最大的问题在于，scipy 包装的是 fortran 77 的优化路径
- 在 fortran 77 的优化路径当中，其整体只使用了双精度，并对于各种存在的精度问题没有做任何优化
- 这使得在实际使用中，scipy 几乎永远不会得到正确的结论，因为各种numerical issue 都会出现，并且不能修复
- 这也是为什么在科学计算中，大部分人还是使用 matlab 的原因

从 scipy 的问题回来，其实在之前的讲解中，我们还有一个问题没有解答，那就是可识别性的问题

什么叫可识别性呢？考虑以下问题

**思考题：请问以下模型是否可以正常优化求解**

- 假设我们的目标是$y$， 我们有 $x_1,x_2,x_3$ 三个变量，并且 $x_3 = 2x_1 + x_2$
- 我们是否能找到 $\beta_1, \beta_2, \beta_3$使得 $\sum_i (y_i - \beta_1x_{i1} - \beta_2 x_{i2} - \beta_3 x_{i3})^2$ 最小
- 如果可能，我们能找到多少个？

**以上问题称之为不可识别性的问题**

- 简单来说，对于一个模型来说，存在（潜在）无穷多个解使得该模型对应的损失函数最小
- 对于存在线性表达式的模型来说，这种情况是极其麻烦的，这里面一种很常见的情况，称之为多重共线性，指的是一些变量可以用其他变量的线性组合表达出来
- 对于 R 来说，这些情况一般可以自动处理，即找到最大线性无关组，很不幸的是，在 python 中，scipy 的实现极烂（大约比正常 C++ 实现慢100 万倍）
- 我们不会对具体算法进行讲解，具体算法已经在第二章当中的 cython例子当中给出，大家可以直接使用



**思考题：one-hot 编码输入逻辑回归之后是否可以正常求解？**

**One-hot 和常数项之间的关系**

- 如果有常数项的话，那么 one-hot 是不可以加入的，原因在于 one-hot编码加起来等于 1
-  如果没有任何常数项的话（以及其他输入是可以的）
-  为什么要加常数项：假设我们用“受教育年限”对“工资”做回归，如果我们我们不加常数项，则等于我们认为未受过的教育的人的工资应该是 0，这显然是不符合实际的

## 关于 Tobit 模型的推导
见Cameron and Trivedi (2005) 16.3 的推导和实现



# Proximal Methods和实现

**Proximal Methods 的实现**

- 这一章，我们将会模仿真实的学习过程，及我们根据知名大学写出来的讲义，直接尝试实现
-  我们的目标是实现对数似然函数加上 $I_1$ 损失的情况
-  在这里，假设对数似然函数为 $l_{\beta} (X, y)$， 其中 $\beta$ 为待估参数，而 $X, y$ 为数据，我们的目标是最小化 $−I_{\beta}(X, y) + \lambda \|\beta \|_1$，其中 $\lambda ≥ 0$ 为惩罚参数
-  具体实现过程首先由我们进行练习，然后再进行实现

**实现细节**
见 Colab Notebook

**思考题（进阶）：如何提升模型的效果**

- 在上面的学习中，我们使用的 step size 都是固定的；在这种情况，我们的结果类似于梯度下降；
-  那么是否有办法采用不同的 step size 呢？

**如何对自己的算法 debug**

- 首先务必检查数学推导是否正确：一般来说，最好的方法是和其他材料做交叉验证
-  其次务必检查每一步是否都有合适的结果
-  最后运行整个算法的时候，需要注意：
  - 算法是否真的收敛了？
  - 是否有 overflow 和 underflow?
  - 在多大情况下，算法会运行到一个局部最优？
  - 是否可以通过调整初始值的方法加速收敛？
  - 是否可以改变 line_search 的方向？

**关于 Proximal Methods 的一些应用的说明**

- Proximal methods 主要应用在 l 1 正则化的函数估计上；
-  这类方法还有很多，例如Efron et al. (2004) 和Garrigues and Ghaoui(2008) 等。目前在深度学习上也开始出现应用 (Yun, Lozano, and Yang 2020)。







# 补充习题

## 基本要求
- 实现时间为 24 小时
- 可通过任何一种优化方法（BFGS 或 Proximal Methods）实现
- 根据模型内容，选择任何一种编程语言实现 100 万次以上模拟，并且根据该模拟研究该算法在不同情况下的可靠程度

## 第一题：非参数 kernel 回归
- 请选择至少两种不同的和 y 存在非线性关系的 X 进行实验
- 请实现逻辑回归中的 Kernel Regression 方法，见Cameron and Trivedi(2005) 第 9.5。并实现 Monte Carlo 估计
- 请回答：
  - 不同的 bandwidth 对于问题的影响有多大
  - 当 X 之间的相关性增加时，估计量效果如何？

## 第二题：Bayesian MCMC 估计
- 请复现Cameron and Trivedi (2005) 的 11.36 的内容
- 请研究 Prior 在样本增加时对于 Posterior 的影响大小

## 第三题：Nested Logic
- 阅读Cameron and Trivedi (2005) 的第 15.6 节，并实现 Nested Logic 模型的估计
- 研究如果 Nested Structure 有问题时候，上一层估计量的影响

## 第四题：Ordered Regression
- 阅读Cameron and Trivedi (2005) 的 15.9.1 节，并实现该模型
- 研究如果 ϵ 来自于和 log-likehood 不同的分布时，估计量的性质

## 第五题：Tobit 模型

- 阅读Cameron and Trivedi (2005) 的 16.3 节，并实现该模型
- 检查当 ϵ 为柯西分布时对整个估计的影响

 ## 第六题：Roy 模型
- 阅读Cameron and Trivedi (2005) 的 16.7 并实现 Roy Model
- 检查当 16.47 式子中，当 σ 假定有错误的情况下，对于 Roy Model 的估计有什么影响

## 第七题：Survival Analysis
- 阅读Cameron and Trivedi (2005) 的 17.6 节并且实现
- 检查在 Hazard Function 指定错误的情况下模型的表现

## 第八题：Finite Mixture of Count Regress
- 阅读Cameron and Trivedi (2005) 的 24.3 节，并实现模拟
- 请检查当 latent class 数量指定错误时候，模型的结果

## 第九题：Censored Count Regression

- 阅读Cameron and Trivedi (2005) 的 24.4 节，并实现 truncation 和censored 中任选一种模型
- 请检查当 truncation 或者 censoring 错误时候，其估计结果的正确性





## 多重共线性

多重共线性是指**线性回归模型**中的解释变量之间由于存在精确相关关系或高度相关关系而使模型估计失真或难以估计准确。

一般来说，由于经济数据的限制使得模型设计不当，导致设计矩阵中解释变量间存在普遍的相关关系。完全共线性的情况并不多见，一般出现的是在一定程度上的共线性，即近似共线性。

### 简介

对线性回归模型
$$
Y = \beta_0 + \beta_1X_1 + \cdots + \beta_pX_p + \epsilon
$$
基本假设之一是自变量， $X_1,X_2,\cdots,X_p$ 之间不存在严格的线性关系。如不然，则会对回归参数估计带来严重影响。为了说明这一点，首先来计算线性回归模型参数的 LS 估计的均方误差。为此。重写线性回归模型的矩阵形式为
$$
Y = \beta_01 = X\beta + \epsilon
$$
其中 $\epsilon$ 服从多元正态分布 $N(0, \sigma^2 I_n), 1 = (1,....,1)^T$，设计矩阵$X$是$n \times p$的，且秩为 $p$。这时，参数$\beta_0$的LS估计为$\hat{\beta_0} = \bar{Y} = \frac{1}{n}\sum_{i=1}^{n}y_i$，而回归系数的LS估计为$\hat{\beta}=(\hat{\beta_1},\cdots,\hat{\beta_p})^T = (X^TY)^{-1}X^TY$。注意到由此获得的LS估计是无偏的，于是估计$\hat{\beta}$的均方误差为：
$$
MSE(\hat{\beta}) = E(\hat{\beta} - \beta)^T(\hat{\beta} - \beta) = \sigma^2\sum_{I=1}^p \frac{1}{\lambda_i}
$$
其中 $\lambda_1 \geq, \cdots, \lambda_p \geq 0$ 的特征根。显然，如果$(X^TX)$至少有一个特征根非常接近于零，则 $MSE(\hat{\beta})$ 就很大，$\hat{\beta}$也就不再是$\beta$ 的一个好的估计。由线性代数的理论知道，若矩阵$(X^TX)$的某个特质根接近零，就意味着矩阵 $X$ 的列向量之间存在近似线性关系。

如果存在一组不全为零的数 $\alpha_1,\alpha_2,...,$，使得

$$
\alpha_1 X_{i_1} + \alpha_2X_{i_2} + ... + \alpha_rX_{i_r} = 0
$$
则称线性回归模型存在完全共线性；如果还存在随机误差 $v$ ，满足 $Ev = 0, Ev^2 < \infty$，使得
$$
\alpha_1 X_{i_1} + \alpha_2X_{i_2} + ... + \alpha_rX_{i_r} + v = 0
$$
则称线性回归模型存在非完全共线性。

如果线性回归模型存在完全共线性，则回归系数的 LS 估计不存在，因此，在线性回归分析中所谈的共线性主要是非完全共线性，也称为复共线性。判断复共线性及其严重程度的方法主要有特征分析法(analysis of eigenvalue)，条件数法 (conditional numbers)和方差扩大因子法(variance inflation factor)。

**产生原因**

主要有3个方面：

（1）经济变量相关的共同趋势

（2）滞后变量的引入

（3）样本资料的限制

**影响**

（1）完全共线性下参数估计量不存在

（2）近似共线性下OLS估计量非有效

多重共线性使参数估计值的方差增大，1/(1-r2)为方差膨胀因子(Variance Inflation Factor, VIF)如果方差膨胀因子值越大，说明共线性越强。相反 因为，容许度是方差膨胀因子的倒数，所以，容许度越小，共线性越强。可以这样记忆：容许度代表容许，也就是许可，如果，值越小，代表在数值上越不容许，就是越小，越不要。而共线性是一个负面指标，在分析中都是不希望它出现，将共线性和容许度联系在一起，容许度越小，越不要，实际情况越不好，共线性这个“坏蛋”越强。进一步，方差膨胀因子因为是容许度倒数，所以反过来。

总之就是找容易记忆的方法。

（3）参数估计量经济含义不合理

（4）变量的显著性检验失去意义，可能将重要的解释变量排除在模型之外

（5）模型的预测功能失效。变大的方差容易使区间预测的“区间”变大，使预测失去意义。

需要注意：即使出现较高程度的多重共线性，OLS估计量仍具有线性性等良好的统计性质。但是OLS法在统计推断上无法给出真正有用的信息。

**解决方法**

（1）排除引起共线性的变量

找出引起多重共线性的解释变量，将它排除出去，以逐步回归法得到最广泛的应用。

（2）差分法，时间序列数据、线性模型：将原模型变换为差分模型。

（3）减小参数估计量的方差：岭回归法（Ridge Regression）。

（4）简单相关系数检验法





