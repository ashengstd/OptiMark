# 线性空间
## 向量表示
使用符号 $\mathbb R$ 表示实数集合, 用 $\mathbb R_n$ 表示$n-$维实向量的集合,  一个实向量可以写成：
$$x=\begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \\ \end{pmatrix}$$
当需要使用行向量时, 我们用向量$x$的转置$x^T = ({x_1},{\cdots},{x_n})$。

## 线性空间
### Def
集合 $V$ 是实数域 $R$ 上的向量空间 (线性空间), 如果 $V$ 中定义如下两种运算:
- 向量加法 **(vector addition)** $+$, 给定两个向量  $v,w \in V$, 向量 加法对应的向量仍包含于$V$中, $v+w \in V$
-  数量乘法 **(scalar multiplication)**$\times$, 对于 $v \in V$ 和实数 $\alpha \in \Bbb R$ , 对应的向量 $\alpha \in \Bbb R$。 

### 一些重要的线性空间:
- 所有由 $m\times n$ 矩阵构成的集合 $\Bbb R^{m \times n}$ 在矩阵加法和数值-矩阵乘法下，构成线性空间。 
$$\begin{align} &A=a_{ij},B=b_{ij} \\ 
&A+B = a_{ij}+ b_{ij}=c_{ij} \\
\end{align}
$$
- 所有定义在$\left[ {a}，{b} \right]$ 上的连续函数构成的集合
$$C = \{f\mid f:\left[{a},{b}\right]\rightarrow{\Bbb R},f\mbox{ is continuous}\}$$
> 可以验证，对于 $f,g \in C$ 和实数$\alpha$ , 有 $f+g \in C, \alpha f \in C$, 因此连续函数的空间是一个向量空间. 

## 线性独立
一组向量的线性组合可以表示为：
$${\alpha_1}{\mathbf v_1}+{\alpha_2}{\mathbf v_2}+{\cdots}+{\alpha_k}{\mathbf v_k},{\alpha_i} \in \Bbb R, {\mathbf v_i} \in \Bbb {R^n}$$
### Def
对于一组向量${\mathbf v_1},{\mathbf v_2},\cdots,{\mathbf v_k}$, 如果线性组合 ${\textstyle \sum_{i=1}^{k}} {\alpha_i}{\mathbf v_i}=0$ 当且仅当 ${\alpha_1 = 0},\cdots,{\alpha_k=0}$ 时成立，那么称向量组${\mathbf v_1},{\mathbf v_2},\cdots,{\mathbf v_k}$ 为线性独立。
一组向量如果不是线性独立的，那么存在非零参数 $\{\alpha_{i} \}$ ，使得 ${\textstyle \sum_{i=1}^{k}} {\alpha_i}{\mathbf v_i}=0$ 。此时我们称向量组 $\{ {\mathbf v_1},{\mathbf v_2},\cdots,{\mathbf v_k}\}$ 线性相关。
### Fact
给定线性独立向量${\mathbf v_1},{\mathbf v_2},\cdots,{\mathbf v_n} \in \Bbb R^n$, 都存在 $\alpha_i,i=1,\cdots,n$ 使得
$$x=\alpha_1 \mathbf v_1+\alpha_2 \mathbf v_2+\cdots+\alpha_n \mathbf v_n$$
## 子空间
### Def 
向量空间$\Bbb V$的子集 $\mathit{S} \subset V$ 是 $V$ 的子空间，如果它本身也是一个向量空间，即对于任意 $\mathbf{v_1,v_2} \in \mathit{S},\alpha \beta,\in \Bbb R$ ，都有$$\alpha\mathbf{v_1}+\beta\mathbf{v_2}\in\mathit S$$
### 例
任何经过原点的直线都是一个子空间。 对于任何非零向量 $\mathbf{v}\in\Bbb{R}^n$ ,过原点的以 $\mathbf{v}$ 为方向向量的直线都可以表示为$$\{\mathbf{x}\in\Bbb{R}^n|\mathbf{x}=t\mathbf{v},t\in\Bbb{R},\mathbf{v}\in\Bbb{R}^n,\mathbf{v}\not=0\}$$
- 任意子空间必定包含零点：$0\in\mathit{S}$
- 子空间是通过原点的直线，平面的推广。
- 任意子空间都可以表示为由一组向量张成的向量空间。
$$\mathbf{S}=span(\mathbf{x}_1,\ldots,\mathbf{x}m):=\Bigg\{
\sum_{i=1}^{n}\lambda_i\mathbf{v}_i|\lambda_i\in \Bbb{R}\Bigg\}$$
## 向量空间的自然基
### Def
一组向量 $\mathbf{v_1,v_2,\cdots,v_k}$ 称为向量空间 $V$ 的一组基，如果它是最大的线性无关集合，即对于任意的 $x\in V$ ，都存在系数 $\lambda_1,\ldots,\lambda_n$，使得 $\mathbf{x}$ 可以由此向量组线性表示
$$\mathbf{x}=\sum_{i=1}^{n}\lambda_i\mathbf{v}_i$$
### 自然基
由向量 $\mathbf{e}_i$ 构成的向量组，其中 $\mathbf{e}_i$ 仅第 $i$ 分量为1，其余均为0：
$$\mathbf{e}_i=(0,\ldots,1,\ldots,0)$$
例如：对于 $\Bbb{R}^3$ ,有
$$\mathbf{e_1}=\begin{pmatrix}1 \\ 0 \\ 0 \end{pmatrix},\mathbf{e_2}=\begin{pmatrix}0 \\ 1 \\ 0 \end{pmatrix},\mathbf{e_3}=\begin{pmatrix}1 \\ 0 \\ 1 \end{pmatrix}$$
## 子空间的基
### Def
对于给定子空间 $\mathcal{S} \subseteq \Bbb{R}^n$ ,它的一组基是指能够张成 $\mathcal{S}$ 的一组独立向量，对于 $\mathcal{S}$ 的基向量 $\{\mathbf{u_1,\ldots,u_r}\}$ , $\mathcal{S}$ 中任意向量均可有由 $\mathbf{u}_i$ 的线性组合表示，即：
$$\mathbf{x}=\sum_{i=1}^{r}\lambda_i\mathbf{u}_i$$
### 维度
对于给定向量空间，基向量的个数不依赖于基向量的选择，这被称为此向量空间的维度，相应的，子空间的维度由子空间中独立向量的数目给定。
例如：任意过原点的直线都构成一个子空间。
## 仿射子空间（Affine Subspace）
### Def
对于给定子空间 $\mathcal{S}\subseteq\Bbb{R}^n$ 和任意 $x_0\in\Bbb{R}^n$ ，集合
$$x_0+\mathcal{S}=\{x_0+x\in\Bbb{R}^n| x \in \mathcal{S}\}$$
称为仿射子空间。
例子：$\Bbb{R}^n$ 中的 $x_0$ 的以 $\mathbf{u}$ 为方向向量的直线为
$$\{x=x_0+t{\mathbf{u}}|t\in\Bbb{R}\}$$
# 内积与范数
相比一般线性空间，内积空间上通过内积定义了向量之间的数量关系，从而能够更好的进行向量之间的运算。
## 内积
### Def
映射 $\langle \cdot,\cdot \rangle:\Bbb{V\times V\rightarrow F}$ ，称为向量空间上的内积，如果满足以下性质：
- 线性性（Linearity）：$\langle\alpha\mathbf{u}+\beta\mathbf{v},\mathbf{x}\rangle = \alpha\langle\mathbf{u,x}\rangle+\beta\langle\mathbf{v,x}\rangle$ 
- 对称性（Symmetry）：$\langle\mathbf{u,v}\rangle=\langle\mathbf{v,u}\rangle$
- 正定性（Positive Definiteness）:$\langle\mathbf{u,v}\rangle\geq0,\langle\mathbf{u,v}\rangle=0\iff \mathbf{v}=0$ 
此时称 $(\Bbb{V},\langle\cdot,\cdot\rangle)$ 为内积空间。
> 内积是向量空间上的一种结构。

## 欧式空间
### Def
对于实向量空间 $\Bbb{R}^n$ ，定义两个向量 $\mathbf{x,y}\in\Bbb{R}^n$ 之间的内积 $\mathbf{x}^{T}\mathbf{y}$：
$$\langle\mathbf{x,y}\rangle=\mathbf{x}^{T}\mathbf{y}=\sum_{i=1}^n{x_i}{y_i}=\mathbf{y}^T\mathbf{x}$$
此内积空间称为欧式空间，常记为 $\Bbb{R^n}$。
> 注：欧式空间中通常根据上下文将 $\mathbf{x}\in\Bbb{R}^n$ 称为“点”或者“向量”。一般的，当我们强调位置时，将其称为点；强调方向时，称为向量。

正交：两个向量 $\mathbf{x,y}\in\Bbb{R}^n$ 称为相互正交的，如果其内积为零 $\mathbf{x}^T\mathbf{y}=0$ .
## 向量范数（Norm）
向量范数是对向量“长度”的刻画。
### Def
令 $V$ 为线性空间/向量空间，$\Vert\cdot\Vert:V\rightarrow\Bbb{R}$ 为一个映射，如果映射  $\Vert\cdot\Vert$ 满足以下性质，则称其为一个范数。
- 正定性：对于任意 $\mathbf{x}\in V$ 都有 $\Vert\mathbf{x}\Vert\geq 0$ ，并且 $\Vert\mathbf{x}\Vert\iff\mathbf{x}=0$  
- 齐次性：对于任意 $\mathbf{x}\in V$ 及 $\alpha\in\Bbb{R}$ ，都有 $\Vert\alpha\mathbf{x}\Vert=\vert\alpha\vert\Vert\mathbf{x}\Vert$ 
- 三角不等式：$\Vert\mathbf{x+y}\Vert\leq\Vert\mathbf{x}\Vert+\Vert\mathbf{x}\Vert$  

范数给出了度量向量“方式”的一种方式
任何内积都能引诱一种对应的范数，对于内积空间 $(\Bbb{V},\langle\cdot,\cdot\rangle)$ ，可以相应定义
$$\Vert\mathbf{x}\Vert=\sqrt{\langle\mathbf{x},\mathbf{x}\rangle}$$
> 此定义满足范数的性质，构成一个范数空间，此范数被称为由内积诱导的范数。

## 欧氏范数（Euclidean Norm）：
### Def
向量的欧氏范数又称为 $\ell_2$ 范数, 定义为:
$$\Vert\mathbf{x}\Vert_2:=\sqrt{\mathbf{x}^T\mathbf{x}}=\sqrt{\sum_{i=1}^n{x_i^2}}$$
欧氏范数是最常用的范数，因为欧氏范数与标准的向量内积相容，即对于任何向量 $\mathbf{x}\in\Bbb{R}^n$，其欧氏范数的平方即为向量内积。
根据定义, 满足 $\Vert\mathbf{x}\Vert_2=1$ 的点构成的集合为一个单位球面。
$$\{\mathbf{x}\in\Bbb{R}^n|\Vert\mathbf{x}\Vert=1\}=\{\mathbf{x}\in\Bbb{R}^n|\sum{x_i^2}=1\}$$
当 $n=2$ 时, 为单位圆.
## $\ell_p$-范数
$$\Vert\mathbf{x}\Vert_p=\Bigg(\sum_{i=1}^{d}{\vert x_i\vert^p}\Bigg)^\frac{1}{p}$$
验证, 此定义满足范数定义, 称为 $\ell_p$ 范数。
- $\Vert\mathbf{x}\Vert_0=\textstyle\sum_{i=1}^{d}\mathbb{1}\{x_i\not=0\}$
- $\Vert\mathbf{x}\Vert_0=\textstyle\sum_{i=1}^{d}\vert x_i\vert$
- $\Vert\mathbf{x}\Vert_0=\sqrt{\textstyle\sum_{i=1}^{d}x_i^2}$
- $\Vert\mathbf{x}\Vert_\infty=max\{\vert x_i\vert\}$
## The Cauchy-Schwarz Inequality
### Content
对于两个向量 $\mathbf{x,y}\in\Bbb{R}^n$ ，有
$$\mathbf{x}^T\mathbf{y}\leq\Vert\mathbf{x}\Vert_2\cdot\Vert\mathbf{y}\Vert_2$$
其中等式成立当且仅当 $\mathbf{x,y}$ 共线。
### Proof
Cauchy-Schwarz 不等式
$$\mathbf{u}^T\mathbf{v}\leq\Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert$$
其更常见形式为：
$$\Bigg(\sum_{i=1}^{n}u_i v_i\Bigg)^2\leq\Bigg(\sum_{i=1}^{n}{u_i^2}\Bigg)\Bigg(\sum_{i=1}^{n}{v_i^2}\Bigg)$$
构造 

$f(t)=\Vert\mathbf{u}t+\mathbf{v}\Vert^2$ ，则有
$$
\begin{eqnarray}
f(t)&=&\Vert\mathbf{u}t+\mathbf{v}\Vert^2 \\ 
&=& (\mathbf{u}t+\mathbf{v})^T(\mathbf{u}t+\mathbf{v})\\
&=& \Vert\mathbf{u}\Vert^2t^2+2\langle\mathbf{u,v}\rangle t+\Vert\mathbf{v}\Vert^2
\end{eqnarray}
$$
因此，$f(t)\leq0$，由此可得$\Delta=4\vert\langle\mathbf{u,v}\rangle\vert^2-4\Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert\leq0$
$$\mathbf{u}^T\mathbf{v}\leq\Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert$$
等价于：
$$-1\leq\frac{\mathbf{u}^{T}\mathbf{v}}{\Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert}\leq1$$
因此, 可以定义两个向量 $\mathbf{u,v}$ 之间的夹角为
$$\cos(\theta)=\frac{\mathbf{u}^{T}\mathbf{v}}{\Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert}$$
故，$\mathbf{u}^T\mathbf{v}=\Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert\cos(\theta)$
对于非零向量 $\mathbf{x,y}\in\Bbb{R}^n$ 其夹角 $\theta$ 为：
$$\cos(\theta)=\frac{\mathbf{x}^{T}\mathbf{y}}{\Vert\mathbf{x}\Vert\Vert\mathbf{y}\Vert}=\Big(\mathbf{\frac{x}{\Vert x\Vert}}\Big)^T\Big(\mathbf{\frac{y}{\Vert y\Vert}}\Big)$$
### 性质
由上式可知: 
- 根据 Cauchy-Schwartz 不等式及其等式成立条件，可知上式的取值 范围为$[-1,1]$。
- 上式是平面上两个向量夹角的一般形式，与二维向量的夹角相一致
- 当两个向量相互正交 $\mathbf{x}^{T}\mathbf{y}=0$，有$\theta=\frac{\pi}{2}$
### 基本恒等式
$$\mathbf{u}^T\mathbf{v}=\frac{1}{2}(\Vert\mathbf{u}\Vert^{2}+\Vert\mathbf{v}\Vert^{2}-\Vert\mathbf{u-v}\Vert^2)$$
可由下面得到：
$$
\begin{eqnarray}
\Vert\mathbf{u-v}\Vert^{2}&=&(\mathbf{u-v})^T(\mathbf{u-v}) \\
&=& \mathbf{u}^{T}\mathbf{u}+\mathbf{v}^{T}\mathbf{v}-\mathbf{u}^{T}\mathbf{v}-\mathbf{v}^{T}\mathbf{u} \\
&=& \Vert\mathbf{u}\Vert^{2}+\Vert\mathbf{v}\Vert^{2}-2\mathbf{u}^T\mathbf{v}
\end{eqnarray}
$$
此恒等式可以称为: **Fundamental Theorem of Optimization**. 根据前述向量夹角定义, 有：
$$\Vert\mathbf{u-v}\Vert^{2}=\Vert\mathbf{u}\Vert^{2}+\Vert\mathbf{v}\Vert^{2}-2\Vert\mathbf{u}\Vert\Vert\mathbf{v}\Vert\cos(\theta)$$
## 投影
### Def
过点  $x_0\in\Bbb{R}^n$ 的以 $\mathbf{u}\in\Bbb{R}^n$ 为方向向量的直线方程为
$$L=\{x_0+t{\mathbf{u}}|t\in\Bbb{R}\}$$
向量在直线的投影: 点 $x$ 在直线上的投影是点 $x$ 到直线上距离最近的 点 (在欧氏范数意义上). 这对应于如下问题: 
$$
\min_{t}\Vert{x-(x_0+t\mathbf{u})}\Vert
$$
这是欧氏空间中向一般集合投影的例子, 也是最优化中一类普遍问题即 最小二乘法一个具体例子.

假设 $\mathbf{u}$ 为归一化变量，即 $\Vert\mathbf{u}\Vert_{2}=1$ ，上述投影问题的目标函数的平方为${\Vert\mathbf{x-x_{0}}-t\mathbf{u}\Vert}^2_{2}=t^2-2t\mathbf{u}^T(\mathbf{x-x_0})+\Vert\mathbf{x-x_{0}}\Vert^2_{2}=(t-\mathbf{u}^T(\mathbf{x-x_{0}}))^2+\mbox{constant}$ 因此，投影问题的最优解为：
$$
t^*=\mathbf{u}^T(\mathbf{x-x_{0}})
$$
相应的投影向量为：
$$
\mathbf{z}^*=\mathbf{x_{0}}+t^*\mathbf{u}=\mathbf{x_{0}}+\mathbf{u}^T(\mathbf{x-x_{0}})\mathbf{u}
$$
其中，$\mathbf{u}^T(\mathbf{x-x_{0}})$ 是 $\mathbf{x-x_{0}}$ 沿 $\mathbf{u}$ 方向的向量。
注意, 这里 $\mathbf{u}$ 是单位向量. 如果 $\mathbf{u}$ 不是单位向量, 可以用 $\mathbf{u}/\Vert\mathbf{u}\Vert_{2}$ 代替 $\mathbf{u}$ , 其对应的投影向量为：
$$
\mathbf{z}^*=\mathbf{x_{0}}+\frac{\mathbf{u}^T(\mathbf{x-x_{0}})}{\mathbf{u}^T\mathbf{u}}\mathbf{u}
$$
### 几何意义
几何意义: 如果$\mathbf{u}$是单位向量$(\Vert\mathbf{u}\Vert_{2}=1)$, 则向量 $\mathbf{x}$ 沿向量 $\mathbf{u}$ 的投影 (即沿 $\mathbf{L}$ 的投影) 为 $\mathbf{z}^*=(\mathbf{u}^T\mathbf{x})\mathbf{u}$  , 其长度为 $\Vert\mathbf{z}^*\Vert=\vert\mathbf{u}^T\mathbf{x}\vert$。
因此, 内积 $\mathbf{u}^T\mathbf{x}$ 即表示 $\mathbf{x}$ 沿向量 $\mathbf{u}/\Vert\mathbf{u}\Vert_{2}$ 的分量。
## 线性映射
向量空间之间的线性映射指的是保持线性运算的映射。
### Def
令 $V,W$ 为向量空间, 映射 $L:V\to W$ 称为线性映射, 如果对任意 $\alpha,\beta\in\Bbb{R}$ 和 $\mathbf{u,v}\in V$
$$
L(\alpha \mathbf{v}+\beta \mathbf{u})=\alpha L(\mathbf{v})+\beta L(\mathbf{u})
$$
给定向量空间 $V,W$ 的基, 线性映射可以用矩阵表示。
一个线性空间到自身的线性映射常称为线性变换 (linear transformation)。
例
-  $f:\Bbb{R}^n \to \Bbb R,f(x)=\mathbf{a^T x}$ 。证明映射 $f$ 为线性映射. 反之, 亦成立。
-  $tr:\Bbb{R}^{n\times n}\to R$为矩阵迹. 证明 $tr(\cdot)$ 是一个矩阵集上的线性映射。
## 正交矩阵 （Orthogonal matrices）
### Def
如果矩阵 $\mathbf{U}$ 满足 $\mathbf{U}^T\mathbf{U}=I_{n}$ 和 $\mathbf{U}\mathbf{U}^T=I_{n}$ , 则 $\mathbf{U}$ 称为正交矩阵. 如果 $\mathbf{U}=[\mathbf{u_{1},\dots,u_{n}}]$是正交矩阵, 则
$$
\mathbf{u}_{i}^T\mathbf{u}_{j}=\delta_{ij}=\Bigg\{
\begin{eqnarray}
&1&,if\;i=j, \\
&0&, \mbox{otherwise} 
\end{eqnarray}

$$
正交矩阵的逆矩阵为 $\mathbf{U}^{-1}=\mathbf{U}^T$ 。
几何上, 正交矩阵对应于旋转和反射。
### 性质
正交矩阵的几何意义: 正交矩阵对应空间中的旋转和反射. 在正交变换 下, 向量的长度 (欧氏范数下) 和夹角保持不变.