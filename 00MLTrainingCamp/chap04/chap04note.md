[TOC]

# 怎样学数学

**数学的两种学法**

- 把数学当做一门语言来学习：不管它的意思，严格按照要求 $\rightarrow$ 我们主要讲方法
- 数学真正的学法，是以证明为目的的。

**数学的几个层次**

- 对于熟悉的方法能够解出来；

- 对于新的概念能够迅速根据常用结题方法解出来；

- 对于新的概念能够通过一定的trick解出来；

- 对于新的概念可以通过非常完整的解题策略逐渐探索出来；

- 构建新的概念并进行探讨；


目标是第二层

**数学真正的思考方式**

核心：

- Frame and Hypotheses
- Elements and Relationships
- Patterns
- Intuition
- Retrospect and Empathetic
- Bucket(In/Out/New)
- Strategic minds

**学习数学的误区**

- 一定要理解概念的直观含义：很多概念的含义是逐渐在应用中才逐渐“理解” 的，直觉有时候会误导；
- 没有策略的解题：解题前最好想好策略；
- 认为数学的难度是均衡的：事实上，对于同一个概念，不同的人的理解速度是不同；
- 跳到自己感兴趣的地方：数学是有前置知识铺垫的；
- 注意：不同的教材对概念的定义不同，一定要根据实际材料的定义来理解。


**学习数学对机器学习的好处** 

- 更好地理解机器学习领域中的概念，为快速学习新概念打基础；
- 看到问题的本质；
- 容易对算法（尤其深度学习部分）进行创新。


**数学理论的主要内容**

- 机器学习的各种角度和建模流程；
- 概率论和统计学基础概念复习；
- 极大似然体系和 EM 算法；
- 贝叶斯体系和 Variational Bayes 算法；
- 矩阵代数的基本概念复习和 Tensor 求导。


# 机器学习的各种角度和建模流程

**为什么要掌握各种角度**

- 最终目的：效果好，即准确性高；
- 为了达到最终目的，必须从不同的角度考虑。



**函数逼近的视角**

- 最简单的是视角；
- 目标：给定 $X$ 预测 $y$；
- 假设：存在真实的 $y = f_0(X)$；
- 如果知道 $f_0$ ，那么就不需要做任工作。

- 观测 ${X_i , y_i ; i \in J}$；
- 可以假设 $f ∈ F$ ；
- 目标：给定一个损失函数 $c$, 最小化 $\sum_i c (f(X_i ), y_i)$；
- 这个估计可以称之为$\hat{f}$。

**什么样的$\hat{f}$是好的**


- 最理想状况$\hat{f} = f_0$ 。
- 不可能原因（一）：没有所有的 X 和 y 的组合；
- 不可能原因（二）：$f_0 \notin F$
- 不可能原因（三）：求解$\hat{f}$时候有困难；
- 一个自相矛盾的启示：要找到一个足够大的 $F$ 使它包含 $f_0$ , 并且要求 $F$ 应该足够小使得求解比较容易。

线性函数   示性函数

**随机的世界**

- 本质上来说，世界上是随机的。
- 随机的来源：
  - 缺乏信息 → 最主要的原因（在表格化数据中最为明显）；
  - 测量误差 → 大部分信息都有误差；
    - 比如说年龄 800 岁，收入 400 万亿；
  - 模型误差 → 假设模型形式和现实的差别；
  - 估计误差 → 得到模型过程中造成的误差；
  - 优化误差 → 求解过程中的误差；
  - 评估误差 → 评估本身也存在误差。

**缺乏信息和过拟合问题**

- 假设目标是用身高预测体重；
- 为什么不可以进行插值？

**根本原因**

- 缺乏信息：人有胖有瘦，仅仅给定身高，不可能判断；
- 导致结果；如果要求身高必须解释体重，身高就承担了非理性的要求；
- 相关结果：bias 较大。
- 统计学根本区别于函数逼近的原因：
  - 函数逼近：$y = f_0 (X)$；
  - 统计学 $y = f_0 (X) + \epsilon $ 。



**Bias 和 Variance**

- Bias：话说得很详细，但是很不准；
  - 北京明天下午两点四十分会发生里氏 2.6 级地震；
- Variance：含糊其词，但是很准；
  - 在这个世界上有一天会发生地震；
- 往往存在 Bias 和 Variance 的权衡（它本身的数学理论只是针对回归的）；
- Bias 大：过拟合；
- Variance 大：欠拟合。

**测量误差**

- 往往难以处理；
-  是数据预处理一个重要部分。

**模型假设**

- 假设背景：存在一个真实的模型，但无法知道全部误差，所以模型一定会有损失；
  - 但就该损失函数而言，这个真实的模型一定是预测最好的；
- 现实情况：无法知道真实的模型，所以只能采用一些模型来逼近；
- 一般的模型可能估计方差较大，但如果采用的模型与真实模型接近，则效果应该是最好的。

**估计误差**

- 即使对于同样的模型或问题，也有不同的办法得到模型参数；
  - 极大似然估计和贝叶斯估计；
  - 增强学习中的 Q-learning 和 Policy Gradient；
- 好的方法可以减少其中误差。

**估计问题**

- 求解的过程，就是迭代的过程；
- 迭代是否会收敛是一个重要的问题；
- 在神经网络中尤其明显，但在传统模型中也存在。

**评估问题**

- 因为不知道真实的损失函数（除非有无限多的测试样本），所以必须评估；
-  评估的越多，训练样本就越少 → 出现了交叉验证的概念；
-  注意避免不公平的评估。

**评估误区**

- 重要原则：一定要看评估本身的误差多大，然后决定做法是否有提升。
- 重要提示：
- 越是误差小的领域，需要概率的角度越多；
- 误差大的领域，概率的角度可能作用有限，更应该找可以优化的地方。

**理论的例外：预训练的存在** 

- 从概率理论上来说，预训练不应该有任何帮助：预训练和当前任务无关（？），而且模型表达力没有变。
- 预训练是深度学习最重要发明之一：
  - 例子：从一个字预测出词语和预测情感没关系；
  - 现实：预测词语表示了对语义的理解，所以对预测情感有帮助；
  - 从优化的角度来说：有利于优化。

**还有很多角度**

- 很多问题需要根据具体场景具体分析；
- 重点：从不同角度出发（数学思维）；
- 从不同角度看同一个问题：其他角度的进展也可以帮助解决这个问题。






# 概率论和统计学复习

**概率论简介**

- 概率论是描述随机的语言；
- 概率论分为朴素概率论和公理性概率论；
- 重点讲朴素概率论。

**最简单情况：一维离散**

- 一维离散意味着可以直接讨论概率；
- 一维离散意味着可以假设概率取值只是整数。
- 例子：男 =1，女 =2，未知 =3
  - $P(X < 3) = \cdots$
  - $p(X = 1) = \cdots$
  - $P(X \leq x) = \sum_{i\leq x}p(X = i)$，或者用更标准的写法 $P(X \leq t) = \sum_{x \leq t} p(x)$ 

**连续变量**

- 连续意味着可能性是无限的；
- 还是可以定义 $P(X \leq x)$；
- 但是定义 $p(x)$ 的时候却不合适。
思考：为什么？

## **PDF和CDF**

- 在给定一个连续变量时，只能定义 $P(x \leq m) = \int_{-\infty}^{\infty} p(x) dx$；
- 虽然离散和连续的定义有所不同，但是积分本身就是一种非常复杂的加法；
- $F_X(t) := P(X \leq t)$；就是所谓的概率累积分布函数（Probability Cumulative Distribution Function）；
- $p(x)$ 就是所谓的概率密度函数（Probability Density Function），不是概率值。

**概念：**

- PDF：概率密度函数（probability density function）, 在数学中，连续型随机变量的概率密度函数（在不至于混淆时可以简称为密度函数）是一个描述这个随机变量的输出值，在某个确定的取值点附近的可能性的函数。

- PMF : 概率质量函数（probability mass function), 在概率论中，概率质量函数是离散随机变量在各特定取值上的概率。

- CDF : 累积分布函数 (cumulative distribution function)，又叫分布函数，是概率密度函数的积分，能完整描述一个实随机变量X的概率分布。

**数学表示**

**PDF**

如果X是连续型随机变量，定义概率密度函数为$f_X(x)$，用PDF在某一区间上的积分来刻画随机变量落在这个区间中的概率，即
$$
Pr(a \leq X \leq b) = \int_a^b f_X(x)dx
$$
**CDF**

不管是什么类型（连续/离散/其他）的随机变量，都可以定义它的累积分布函数，有时简称为分布函数。

- 对于连续型随机变量，显然有：

$$
F_X(x) = Pr(X \leq x) = \int_{-\infty}{x} f_X(t) dt
$$

- 那么CDF就是PDF的积分，PDF就是CDF的导数。
- 对于离散型随机变量，其CDF是分段函数，比如举例中的掷硬币随机变量，它的CDF为:

$$
F_X(x) = Pr(X \leq x) \left\{\begin{matrix} 0 & if\ x < 0 \\ \frac{1}{2} & if \ 0 \leq x < 1 \\1 & if \ x \geq 1 \end{matrix} \right.
$$



**PMF**

如果X离散型随机变量，定义概率质量函数为$f_X(x)$，PMF其实就是高中所学的离散型随机变量的分布律,即
$$
f_X(x) = Pr(X = x)
$$
比如对于掷一枚均匀硬币，如果正面令$X=1$，如果反面令$X=0$，那么它的PMF就是
$$
f_X(x) = \left\{\begin{matrix}
\frac{1}{2} \ if \ x \in \{0, 1\} \\ 
 0 \ if \ x \notin \{0, 1\}
\end{matrix}\right.
$$


**习题：CDF 和 PDF 的转换**
指数分布的 PDF 为 $λe^{−λx}$ ，$x \geq 0, \lambda > 0$，求其 CDF。求$P(x \leq t)$
$$
P(x \leq t) = -\int_{-t}^{0}  e^{-\lambda x} d(\lambda x) \\
= e^0 - e^{-\lambda t} \\
= 1 - e^{-\lambda t}
$$


**习题：不同参数之间的转换**
- 假设 $X$ 服从正态分布，即 $p(x) = \frac{1}{\sqrt{2 \pi}}exp(- \frac{x^2}{2})$
- 请问 $Y = aX + b$ 的 PDF 是什么？
- 请问 $Z = X^2$ 的 PDF 是什么？
- $\Phi(x) = \int_{- \infty}^{x}\frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}}dt$


$Y = aX + b$
$$
if \ a = 0, \  P(y \leq t) = ? \\
if \ a > 0, P(y \leq t) = P(ax + b \leq t) = P(x \leq \frac{t - b}{a}) = \Phi(\frac{t - b}{a}) \\
Pr(t) = \frac{d}{dt} P(y \leq t) \\
= \frac{d}{dt} \Phi(\frac{t - b}{a}) ,\qquad m = \frac{t - b}{a} \\
= \frac{d}{dm} \Phi(m) \frac{dm}{dt} \\
= \frac{1}{a} p(m)
$$


$$
\begin{align}
& if \ a < 0  \\
& P(y \leq t)  \\
& = P(aX + b \leq t) \qquad p(X = \frac{t -b }{a}) = 0 \\
& = P(X \geq \frac{t-b}{a}) \\
& = 1 - P(X < \frac{t - b}{a}) \\
& = 1 - P(X \leq \frac{t - b}{a}) \\
& = 1 - \Phi(\frac{t - b}{a})
\end{align}
$$

$Z = X^2$

$$
\begin{align}
& P(Z \leqslant t) \\
& = P(x^2 \leqslant t) \\
& = P(-\sqrt{t} \leqslant x \leqslant \sqrt{t}) \\
& = P(x < \sqrt t) - P(x < - \sqrt t) \\
& = P(x < \sqrt t) - P(x \geqslant \sqrt t) \\
& = P(x \leqslant \sqrt t) - (1 - P(x < \sqrt t)) \\
& = 2P(x \leqslant \sqrt t) - 1
\end{align}
$$

$$
\begin{align}
 \frac{d}{dt} P(Z \leqslant t) &= \frac{d}{dt}(2P(x \leqslant \sqrt t) - 1) \\
 &= 2 \frac{d}{dt} P(x \leqslant \sqrt t)\\
 &= 2\frac{d}{dm}\Phi(m) \frac{dm}{dt}  \qquad m = \sqrt t \\
 &= \frac{1}{\sqrt{2 \pi}} e^{-\frac{t}{2}} t^{- \frac{1}{2}}
\end{align}
$$


**多维情况**

- 以二维为例：$P(X \leq m, Y \leq n) = \int_{-\infty}^{m}\int_{-\infty}^{n}p(x, y)dxdy$
- 对于边际分布 $p(x) = \int p(x, y)dy$；
- 条件概率 $p(x|y) = p(x, y)/p(y)$。

**练习：边际分布**

假如 $p(x, y) = \left \{\begin{matrix}x + y & if \ 0 \leq x \leq 1,0\leq y \leq 1 \\ 0 & otherwise \end{matrix}\right.$，求 $Y$ 的边际分布。

$$
\begin{align}
p(x) &= \int_{-\infty}^{+\infty} P(x, y) dy \\
& = \int_0^1 (x + y) dy \\
& = x + \int_0^1 y dy \\
& = x + \frac{1}{2}
\end{align}
$$
**独立性**

- 一般来说，给定随机变量 $X, \ Y$，两者的联合概率分布不能由各自的概率分布单独计算得出；
-  但是当两者独立时，以下为定价条件：
  - 1.对任何事件 $A,\ B$，我们有 $P(X \in A, Y \in B) = P(X \in A)P(Y \in B)$；
  - 2.$p(x, y) = p(x)p(y)$；
  - 3.$p(x|y) = p(x)$。
-  我们常说 i.i.d，即独立同分布。

**练习：独立变量等价性**

请证明 $p(x, y) = p(x)p(y)$ 等价于 $p(x|y) = p(x)$。

$$
\begin{align}
& 3 \rightarrow 2 \\
p(x, y) &= p(x|y)p(y) \\
&= p(x)p(y) \\
& \qquad \\
& 1 \rightarrow 3 \\
p(x|y) &= \frac{p(x,y)}{p(y)} \\
& = \frac{p(x)p(y)}{p(y)} \\
& = p(x)
\end{align}
$$



## 数学期望
给定一个概率密度函数 $p(x)$, 再给定一个函数 $f(x)$，我们定义他的数学期望（Expectation）为
$$
E_p[f(X)]:= \int f(x)p(x)dx
$$
$$
\begin{align}
& continuous: \quad  E_p[f(x)] = \int_{-\infty}^{+\infty}f(x)p(x) dx       \\
& Discrete: E_p[f(x)] = \sum_{?}f(x)p(x)          \\
& Common\ writing: \\
& E[f(x)] \\
& E_{x\sim p}[f(x)] \\
& px \quad \int fp
\end{align}
$$





**条件数学期望**

给定一个条件概率密度函数 $p(x|y)$， 再给定一个函数 $f(x)$，我们定义他的条件数学期望（Conditional Expectation）为
$$
E_p[f(X)|Y = y] := \int f(x)p(x|y)dx
$$
$$
E_p[f(x)|Y = y] \\
E[f(x)|Y] \\
E^Yf(x)
$$



**期望的常用公式**
$$
\begin{align}
& EX \rightarrow average\ value \\
& variance: \quad var\ X(vX) = E|(X - EX)^2| = EX^2 - (EX)^2 \\  
& Covariance: \quad cov\ (X, Y) = EXY - EXEY
\end{align}
$$


**习题：重期望公式**
$$
E_Y[E_p(f(X))|Y] = E_p[f(X)]
$$

解答
$$
\begin{align}
& E_Y[E_{x\sim p}[f(x)|Y]] \\
& = E_Y[\int f(x)p(x|y)dx] \\
& = \int \int f(x)p(x|y)p(y)dxdy \\
& = \int \int f(x)p(x, y) dxdy \\
& = \int f(x) [\int p(x, y)dy]dx \\
& because\ of\ Marginal\ distribution\ formula\ \int p(x, y)dy = p(x) \\
& = \int f(x)p(x) dx \\
& = E[f(x)]
\end{align}
$$


**习题：证明$X, Y$ 独立， 则 $cov(X, Y) = 0$**

证明：
$$
\begin{align}
& because \ X\ and\ Y\ are\ independent,\ so\ p(x, y) = p(x)p(y) \\
& because\ cov(X, Y) = EXY - EXEY \  so\ proof\ EXY = EXEY \ is \ ok \\
EXY &= \int \int xy p(x, y) dxdy \\
& = \int \int xyp(x)p(y)dxdy \\
& = \int xp(x)dx \int yp(y)dy \\
& = EXEY 
\end{align}
$$
**习题：$X \sim  Unif[-1, 1]; Z = X^2$**

1.$Cov(X, Z) = 0$

2.$X$不独立于$Z$

证明：
$$
\begin{align}
& Cov(x, Z) = EXZ - EXEZ \quad EX = \int_{-1}^1 \frac{1}{2} x dx = 0.\quad Odd\ function \\
& EXZ = EXX^2 = EX^3 \rightarrow \int_{-1}^1 \frac{1}{2} X^3dx = 0. \quad Odd \ function \\
& \Rightarrow Prove\  1.Cov(X, Z) = 0. \\
& P(X \in A, Z \in B) = P(X \in A)P(Z \in B) \\
& P(x \leqslant \frac{1}{2}, X^2 \leqslant \frac{1}{2}) = P(-\frac{\sqrt 2}{2} \leqslant x \leqslant \frac{1}{2}) ?= P(x \leqslant \frac{1}{2})P( Z \leqslant \frac{1}{2})  \\
& P(-\frac{\sqrt 2}{2} \leqslant x \leqslant \frac{1}{2}) = \frac{1}{4} - \frac{\sqrt 2}{4} \\
& P(x \leqslant \frac{1}{2})P( Z \leqslant \frac{1}{2}) = \frac{3}{4} \times \frac{\sqrt 2}{2}  \\
& \Rightarrow P(x \leqslant \frac{1}{2}, X^2 \leqslant \frac{1}{2}) \neq P(x \leqslant \frac{1}{2})P( Z \leqslant \frac{1}{2}) \\
& Prove\ X\ not \  independent \ Y
\end{align}
$$


**习题：贝叶斯公式**
$$
p(y|x) = \frac{p(y)p(x|y)}{\int p(x|y)p(y)dy}
$$

证明：
$$
\begin{align}
& condition\ 1: p(x) = \int_{-\infty}^{+\infty} p(x, y) dy \\
& condition\ 2: p(x|y) = \frac{p(x, y)}{p(y)}    \\
& \int p(x|y)p(y)dy = \int_{-\infty}^{+\infty}p(x, y) dy = p(x) \\
& p(y|x)p(x) = p(x|y)p(y) \\
& p(y|x)p(x) = p(x, y) \\
& p(x|y)p(y) = p(x, y) \\
& therefore \quad  p(y|x) = \frac{p(y)p(x|y)}{\int p(x|y)p(y)dy}
\end{align}
$$




**重点关注**

- Multinomial: $P(X = x_i) = p_i$；

- 正态分布：$p(x) = \frac{1}{\delta \sqrt{2 \pi}} e^{-\frac{1}{2}(\frac{x - \mu}{\delta})^2}$， 其中 $\mu$  $\delta$是参数。这时候，我们常常写成 $X ~ N(\mu, \delta^2)$；

   

**练习：**

假设 $X ~ N(\mu, \delta^2)$，请求出 $E_X[e^{\lambda X}]$。

思路：
$$
\begin{align}
& p(x = \frac{1}{\sqrt{2 \pi}} e^{-\frac{x^2}{2}})  \\
& E_X(e^{\lambda x}) =  \frac{1}{\sqrt{2 \pi}} \int e^{\lambda x} e^{-\frac{x^2}{2}}dx \rightarrow ? \int \frac{1}{\sqrt{2 \pi}} ? e^{\frac{(x - ?)^2}{2?}} dx \cdot M
\end{align}
$$


解答：
$$
\begin{align}
E_X(e^{\lambda x}) &=  \frac{1}{\sqrt{2 \pi}} \int e^{\lambda x} e^{-\frac{x^2}{2}}dx \\
& = \int \frac{1}{\sqrt{2 \pi}} e^{- \frac{x^2 - 2\lambda x}{2}}dx \\
& = \int \frac{1}{\sqrt{2 \pi}} e^{- \frac{x^2 - 2\lambda x + \lambda^2 - \lambda^2}{2}}dx \\
& = \int \frac{1}{\sqrt{2 \pi}} e^{- \frac{(x - \lambda)^2}{2}}e^{\frac{\lambda^2}{2}}dx  \\
& = e^{\frac{x^2}{2}}
\end{align}
$$



# 极大似然估计

## 极大似然估计基本思路

**极大似然估计：例子**

- 考虑最简单的情况，即掷一个不公平的硬币：
- 每一个硬币向上的概率为 $p(x_i )$，用 $y_i = 1$ 记载硬币向上；
- 由此得到硬币向下的概率为 $1 − p(x_i )$, 用 $y_i = 0$ 表示；
- 整体观测到目前情况的概率为 $p(x_i )^{y_i} \times (1 − p(x_i))^{(1−y_i)}$ ，这就是所谓的似然函数；
- 这个形式比较难看，所以不妨取个 log，这就是对数似然函数：$y_i log(p(x_i )) + (1 − y_i ) log(1 − p(x_i ))$。



**思考：什么是好的 p**

- 如果我们知道 p，那什么都不用做；
- 但实际上我们既不知道 p，还想知道什么是一个好的 p；
- 假设只抛一次硬币，思考下面哪个概率更好？
  - 一个估计 p 的似然函数为 0.3；
  - 另一个估计 p 的似然函数为 0.9。

**极大似然函数的基本思想**

- 找到使目前似然函数最大的那个观测；
- 或者由于对数变换是单调变化，找到负的对数似然函数最小的解。

**继续抛硬币…**

- 只抛一次硬币，当然没有任何做推断的价值；
- 现在假设抛 N 次硬币，得到观测 $\{x_i, y_i; i \leqslant N\}$；
- 继续假定每次抛硬币的结果不影响下一次抛硬币的概率分布，即观测独立；
- 则似然函数为 $\prod_ip(x_i )^{y_i} (1 − p(x_i ))^{(1−y_i )}$；
- 连乘带来的问题：因为如果连乘一个 0 到 1 之间的数，得到的乘积会越来越小，特别小的时候，电脑就会出现数值问题（比如说 10 的负十万次方）。

**如何解决数值问题**

- 取个 log 即可：$log(xy) = log(x) + log(y)$；
- 则负的对数似然函数为：$−\sum_ i (y_i log(p(x_i )) + (1 − y_i ) log(1 − p(x_i )))$；
- 这就是 Binary Cross Entropy。



**如何选择 $p(x_i)$ 的形式**

- $p(x_i )$ 长什么样呢？
- 要控制 $p(x_i )$ 取值在 0 到 1 之间；
- 一个常见选择 $p(x_i) = \frac{1}{1 + exp(-f(x_i))}$
- - 如果 $f(x_i ) = \sum_k β_k x_i$ , 其中 $β_k$ 为未知参数（需要求解），则得到了逻辑回归的数学表达形式；
- 注意：这种$f$的函数形式被称为线性函数，近似于多个线性函数组合的函数是最重要的一类函数形式。



**尝试推导…**

- 现在假设有 $y_i$ ，服从期望为 $f(x_i)$ 且方差为 1 的正态分布；
- 也就是说 $p(y_i) = \frac{1}{\sqrt{2\pi}}exp(−(y_i − f(x_i ))^2 /2)$；


**推导**

我们需要的负的对数似然函数等于

$$
- \sum_{i} \log p(y_i) = -\sum_i (-(y_i - f(x_i))^2)/2 + K
$$
其中 $K$ 是一个跟 $f$ 无关的常数，所以这里最小化的距离是$\sum_i(y_i - f(x_i))^2$这就是最小二乘法

方差大是欠拟合

**分类和回归**

- 第一种情况，称之为二分类问题，对应多分类问题也可以进行对应推导；
-  第二种情况，称之为回归问题；
-  即使在监督学习的框架下，还有很多其他类型的问题。

回归：$y_i = f_{\theta}(x_i) + \epsilon_i, \epsilon_i \sim N(0, \delta^2 ?)$

分类：$y_i = \left \{\begin{matrix} 1 & p_{\theta 1}^{\theta} \\ 2 & p_{\theta 2}^{\theta} \\ ... & ... \\ T & p_{\theta T}^{\theta} \end{matrix} \right.$ $\sum y_i log(p^{\theta}(x_i)) + (1 - y_i)log(1 - p^{\theta}(x_i))$

## Tobit模型

**某银行小微快贷额度测算问题**

- 目标：确定小企业的贷款额度。
- 考虑方向：
  - 违规可能性：要把风险控制在一定范围内；
  - 需求：对贷款需求越高的企业应该给更多贷款。
- 第一个问题可以作为分类问题解决；
- 第二个问题不好解决。

放款

- 风险 ---> 评分卡 + 阈值
- 需求



**基本思想$$**

- 虽然观测不到企业的真实需求，但可以假设存在一个真实需求。
- 我们知道实际放款额和实际使用金额，所以存在两种情况：
  - 放款额度大于实际使用金额，这时可以假定实际需求即为实际使用金额；
  - 放款额度等于实际使用金额，这时虽然不知道实际需求，但是知道实际需求一定大于等于放款额度。

分析

- $z_i$：实际放款额度
- $y_i$：提现额度

有 $y_i \leqslant z_i$

- $y_i^*$：真实需求

$$
y_i^* = f_{\theta}(x_i) + \epsilon_i, \epsilon_i \sim N(0, \delta^2)
$$



**模型的基本思路**

- 假设真实需求为 $y^*_i$；
- 进一步假设 $y^∗_i = f(x_i) + ϵ_i$ ，且假设 $ϵ_i$ 为正态分布；
- 银行所给的真实的额度假设为 $y_i$ ；
- 当发生截断时，其似然函数为 $P(y^∗_i \geqslant y_i)$；
- 当不发生截断时，其似然函数为 $p(y_i)$；
- 两者结合，即可以得到估计方式。



**具体模型**
$$
z_i = \left \{\begin{matrix}1 & full \\ 0 & not\ full  \end{matrix}\right. \\
y_i : Loan\ amount \\
y_i^* = f_{\theta}(x_i) + \epsilon_i, \epsilon_i \sim N(0, \delta^2)
$$

$$
\begin{align}
& while \ z_i = 0, \  y_i = y_i^* = f_{\theta}(x_i) + \epsilon_i \\
& ?p(y_i) = \frac{1}{\sqrt{2\pi}\delta?} exp(- \frac{(y_i - f_{\theta(x_i)})^2}{2\delta}) \\
& while\ z_i = 1 \quad P(y_i \geqslant y_i^*)  \\
& = P(y_i \geqslant f_{\theta}(x_i) + \epsilon_i) \\
& = P(\epsilon_i \leqslant y_i - f_{\theta}(x_i))  \\
& = \Phi(\frac{y_i - f_{\theta}(x_i)}{\delta})   \\
& 1 - \Phi(\frac{y_i - f_{\theta}(x_i)}{\delta})^{z_i}p(y_i)^{1 - z_i}
\end{align}
$$

对于任何 $x_i \sim F$，我们可以将他转换成为正太分布

因为 $P(x_i \leqslant F^{-1}(t_i)) = P(F(X_i)\leqslant t) = t$； 所以 $F(x_i) \sim Unif(0,1)$

设正态分布 CDF为 $\Phi$；由均匀分布的特点可得。
$$
P(u_i \leqslant \Phi(t)) = \Phi(t) = P(F(x_i) \leqslant \Phi(t))
$$
结论：$P(\Phi^{-1}(F(x_i)) \leqslant t) = \Phi(t)$

大数定律：
$$
\begin{align}
& 1.x_1,x_2,\cdots,x_n \quad i.i.d \\
& 2.E|x_1| < x \\
& 3.\frac{1}{n} \sum_{i=1}^{n} x_i \underset{\rightarrow}{n}Ex_1 ???
\end{align}
$$
中心极限定理：

## EM算法和HMM

- 通常情况下，在极大似然框架中，如果容易推导出对数似然函数的话，那么求解将会非常容易；
-  但是如果存在隐变量，则推导变得非常困难；
-  在一些情况下，EM 算法是解决隐变量问题的一个非常通用的框架 (现实情况少见)。

**HMM 算法的推导难度**

- HMM 算法的估计方法称之为 Baum-Welch 算法；
- 现场去 “推导” 该算法是不可能的；
- 现场去 “默写” 该算法是有可能的；
- 默写跟数学能力毫无关系。

### EM算法

考虑以下关系：用$I(\theta; X)$ 表示对数似然函数，则：
$$
\begin{align}
I(\theta; X) &= \log p_{\theta}(X) \\
& = \log \int p_{\theta} (X, y) dy \\
& = \log \int \frac{p_{\theta}(X,y)}{p_{\tilde{\theta}}(y|X)} p_{\tilde{\theta}}(y|X) dy \\
& \geqslant \int \log(p_{\theta}(X, y))p_{\tilde{\theta}}(y|X)dy - \int \log(p_{\tilde{\theta}}(y|X))p_{\tilde{\theta}}(y|X)dy \\
& = E_{\tilde{\theta}}[\log p_{\theta}(X, y)|X] - E_{\tilde{\theta}}[\log p_{\tilde{\theta}}(y|X)|X]
\end{align}
$$
其中：

**注意在这里**

- $y$ 是一个隐变量；
- $\tilde{\theta}$ 是当前的估计，目标是通过迭代的方法找到下一步的估计 $\theta$，因为$E_{\tilde{\theta}}[\log p_{\tilde{\theta}}(y|X)|X]$ 跟$\theta$ 没有关系，所以可以忽略；
- 定义 $Q(\theta, \tilde{\theta}) = E_{\tilde{\theta}}[\log p_{\theta}(X,y)|X]$， 则EM算法可以定义为：
  - 计算 $Q(\theta, \tilde{\theta})$；
  - 最大化 $\theta := argmax_{\theta '}Q(\theta ', \tilde{\theta})$

### 隐马尔可夫链

- 假设对于每一个观测 $d$ 可以观测到 $\{X_t^{(d)}, 1 \leq t \leq T\}$；
- 它的概率分布取决于隐变量 $z_t^{(d)}$ 。并且该变量服从马尔可夫性质，因此如果知道 $t − 1$ 的信息，就不需要知道更早的信息，就可以得到 $z_t^{(d)}$ 的概率分布；
- 假设 $X’s$ 和 $z’s$ 都只能取有限多个值。

#### 推导过程

我们有：
$$
P(z, X; \theta) = \prod_{d = 1}^{D}(\pi_{z_1^{(d)}}B_{z_1^{(d)}}(X_1^{(d)}) \prod_{t = 2}^{T}A_{z_{t - 1}^{(d)}z_t^{(d)}}B_{z_t^{(d)}}(x_t^{(d)}))
$$
**在这里**

- $(d)$ 上标表示观测 $d$；
- $\pi_{z_1^{(d)}}$ 为初始分布；
- $A_{z_{t-1}^{(d)}z_t^{(d)}}$ 为转移概率；
- $B_{z_t^{(d)}}(X_t^{(d)})$ 为发射概率。

**对上式取 log之后**
$$
\log P(z, X; \theta) = \sum_{d=1}^{D}[\log \pi_{z_1^{(d)}} + \sum_{t=2}^{T}\log A_{z_{t-1}^{(d)}z_t^{(d)}} + \sum_{t = 1}^{T}\log B_{z_t^{(d)}}(X_t^{(d)})]
$$
**放到 $Q$ 函数中， 假设目前的参数 $\theta^s$：**
$$
\begin{align}
Q(\theta, \theta^s) &= \sum_{z\in Z}\sum_{d=1}^D\log \pi_{z_1^{(d)}}P(z, X;\theta^s) \\
&+ \sum_{z \in Z}\sum_{d=1}^D\sum_{t=2}^{T}\log A_{z_{t-1}^{(d)}z_t^{(d)}}P(z, X, \theta^s) \\
&+ \sum_{z \in Z}\sum_{d=1}^D\sum_{t=1}^{T}\log B_{z_t^{(d)}}(X_t^{(d)})P(z, X;\theta^s)
\end{align}
$$
**加上拉格朗日乘子：**
$$
\begin{align}
\hat{L}(\theta, \theta^s) &:=Q(\theta, \theta^s) - \lambda_{\pi}(\sum_{i=1}^M \pi_i - 1) \\
& - \sum_{i=1}^M \lambda_{A_i}(\sum_{j=1}^M A_{ij} - 1) \\
& - \sum_{i=1}^M \lambda_{B_i}(\sum_{j=1}^N B_i(j) - 1) 
\end{align}
$$

**首先求解** $\pi_i$。
$$
\begin{align}
\frac{\partial \hat{L} (\theta, \theta^s)}{\partial \pi_i} &= \frac{\partial}{\partial \pi_i}(\sum_{z \in Z} \sum_{d=1}^{D} \log \pi_{z_1^{(d)}}P(z,X; \theta^s))-\lambda_{\pi} = 0\\
& = \frac{\partial}{\partial \pi_i}(\sum_{j=1}^{M}\sum_{d=1}^{D}\log \pi_j P(z_1^{(d)}=j,X;\theta^s)) - \lambda_{\pi} = 0 \\
& = \sum_{d=1}^D \frac{P(z_1^{(d)} = i,X;\theta^s)}{\pi_i} - \lambda_{\pi} = 0
\end{align}
$$

**然后**
$$
\frac{\partial \hat{L}(\theta, \theta^s)}{\partial \lambda_{\pi}} = - (\sum_{i=1}^M \pi_i - 1) = 0
$$

**求解，可以得到：**
$$
\begin{align}
\pi_i &= \frac{\sum_{d=1}^D P(z_1^{(d)}=i, X;\theta^s)}{\sum_{j=1}^D\sum_{d=1}^D P(z_1^{(d)} =j, X;\theta^s)} = \frac{\sum_{d=1}^D P(z_1^{(d)}=i,X;\theta^s)}{\sum_{d=1}^D \sum_{j=1}^M P(z_1^{(d)} = j, X; \theta^s)} \\
&= \frac{\sum_{d=1}^DP(z_1^{(d)} = i,X; \theta^s)}{\sum_{d=1}^D P(X;\theta^s)} = \frac{\sum_{d=1}^D P(z_1^{(d)} = i,X; \theta^s)}{DP(X;\theta^s)} \\
&= \frac{\sum_{d=1}^D P(X;\theta^s)P(z_1^{(d)} = i|X; \theta^s)}{DP(X;\theta^s)} = \frac{1}{D}\sum_{d=1}^D P(z_1^{(d)} = i|X; \theta^s)\\
&= \frac{1}{D}\sum_{d=1}^D P(z_1^{(d)} = i|X^{(d)}; \theta^s)
\end{align}
$$

**采用类似方法：**
$$
\begin{align}
A_{ij} &= \frac{\sum_{d=1}^D\sum_{t=2}^T P(z_{t-1}^{(d)}=i, z_t^{(d)}=j,X;\theta^s)}{\sum_{j=1}^M \sum_{d=1}^D\sum_{t=2}^T P(z_{t-1}^{(d)}=i, z_t^{(d)}=j,X;\theta^s)} \\
&= \frac{\sum_{d=1}^D\sum_{t=2}^T P(z_{t-1}^{(d)}=i, z_t^{(d)}=j,X;\theta^s)}{\sum_{d=1}^D\sum_{t=2}^T P(z_{t-1}^{(d)}=i,X;\theta^s)} \\
&= \frac{\sum_{d=1}^D\sum_{t=2}^T P(X;\theta^s)P(z_{t-1}^{(d)}=i, z_t^{(d)}=j|X;\theta^s)}{\sum_{d=1}^D\sum_{t=2}^T P(X;\theta^s)P(z_{t-1}^{(d)}=i|X;\theta^s)} \\
&= \frac{\sum_{d=1}^D\sum_{t=2}^T P(z_{t-1}^{(d)}=i, z_t^{(d)}=j|X^{(d)};\theta^s)}{\sum_{d=1}^D\sum_{t=2}^T P(z_{t-1}^{(d)}=i|X^{(d)};\theta^s)} \\
\end{align}
$$
**更进一步：**
$$
\begin{align}
B_i(j) &= \frac{\sum_{d=1}^D\sum_{t=1}^T P(z_t^{(d)} = i,X; \theta^s)I(x_t^{(d)}=j)}{\sum_{j=1}^N \sum_{d=1}^D\sum_{t=1}^T P(z_t^{(d)}=i, X;\theta^s)I(x_t^{(d)}=j)} \\
&= \frac{\sum_{d=1}^D\sum_{t=1}^T P(z_t^{(d)} = i,X; \theta^s)I(x_t^{(d)}=j)}{\sum_{d=1}^D\sum_{t=1}^T P(z_t^{(d)} = i,X; \theta^s)} \\
&= \frac{\sum_{d=1}^D\sum_{t=1}^T P(z_t^{(d)} = i|X^{(d)}; \theta^s)I(x_t^{(d)}=j)}{\sum_{d=1}^D\sum_{t=1}^T P(z_t^{(d)} = i|X^{(d)}; \theta^s)} \\
\end{align}
$$
**思考一下**

- 为什么要推导 $P(z_{t-1}^{(d)}=i, z_t^{(d)}=j|X^{(d)};\theta^s)$ 和 $P(z_t^{(d)} = i|X^{(d)};\theta^s)$？
- 这是因为这两者可以用动态规划来求解

**难点在哪里？**

$P(z_{t-1}^{(d)}=i, z_t^{(d)}=j|X^{(d)}$ 和 $P(z_t^{(d)} = i|X^{(d)};\theta^s)$可以动态求解有效动态求解这件事情不可能一眼看出来，甚至我们在开始推导的时候也不可能考虑到动态求解的问题。

# 贝叶斯估计和变分贝叶斯

**贝叶斯学派和频率学派**
- 在之前所有的模型中，我们均假设有所谓的真实参数或模型，目的是为了推导出该真实的模型。
- 贝叶斯学派的视角不同：
  - 假设参数是 $\theta$，将会对其有一个 prior，表示为 $p(\theta)$，而 $\theta$ 本身就是随机；
  - 现在得到了观测 $X$，目标是得到 posterior：$p(\theta|X)$。
- 根据贝叶斯公式，有

$$
p(\theta|X) = \frac{p(X|\theta)p(\theta)}{\int p(X|\theta)p(\theta)d\theta}
$$

**举例**

假设 $\mu \sim N(0,1), X|\mu \sim N(\mu,1)$，我们一起来推导 $\mu$ 的 posterior

$$
\begin{align}
p(\mu|X) &\propto exp(-\mu^2/2)\prod_{i=1}^N exp(-\sum_i (X_i - \mu)^2/2) \\
&\propto exp(-(\frac{N+1}{2}\mu^2 -\mu \sum_iX_i)) \\
&\propto exp(-(\mu^2-\frac{2\sum_iX_i}{N+1}\mu)/(\frac{2}{N+1})) \\
&\propto exp((\mu - \frac{\sum_iX_i}{N+1})^2/\frac{2}{N+1}) \\
\end{align}
$$
**结论**

- $\mu|X \sim N(\frac{\sum_iX_i}{N+1}, \frac{1}{(N+1)^2})$；
- 因此，posterior也是正态分布；
- 这称之为Conjugate Priors。

**贝叶斯方法的好处和坏处**

- 好处：
  - 方便处理隐变量；
  - 可以对不确定性进行估计。
- 坏处：计算麻烦 → 就目前的深度学习应用来说，最方便的是变分法，

**证明下式**
$$
\log p(X) - D[q(z)\|p(z|X)] = E_{z\sim q}[\log(pX|z)] - D[Q(z)\|P(z)]
$$

其中 D 为KL-divergence，即$D(P(x)\|Q(x)) = E_{X\sim P}(\log p(X) - \log q(X))$。

证明：
$$
\begin{align}
& l.h.s = \log p(x) - \int \log q(z)q(z)dz + \int \log p(z|X)q(z)dz \\
& r.h.s = \int \log p(X|z)q(z)dz - \int \log q(z)q(z)dz + \int \log p(z)q(z) dz \\
& p(X|z) = p(x,z)/p(z) \\
& r.h.s = \int (\log p(X,z) - \log p(z))q(z)dz  + \int \log p(z)q(z) dz - \int \log q(z)q(z)dz  \\
& r.h.s = \int \log p(X,z)q(z)dz - \int \log q(z)q(z)dz  \\
& r.h.s = \int (\log p(z|X) + \log p(x) )q(z)dz - \int \log q(z)q(z)dz  \\
& r.h.s = \int\log p(x)q(z)dz + \int \log p(z|X)q(z)dz - \int \log q(z)q(z)dz  \\
& r.h.s = \log p(x)\int q(z)dz + \int \log p(z|X)q(z)dz - \int \log q(z)q(z)dz  \\
& \int q(z)dz = 1 \\
& r.h.s = \log p(x) + \int \log p(z|X)q(z)dz - \int \log q(z)q(z)dz = l.h.s \\
\end{align}
$$



