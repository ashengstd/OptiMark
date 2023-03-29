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
## 正交矩阵 （Orthogonal Matrices）
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
令 $\mathbf{x}\in\Bbb{R}^{n}$ ，向量长度
$$
\Vert\mathbf{Ux}\Vert^2_{2}=(\mathbf{Ux}^T)\mathbf{Ux}=\mathbf{x}^T
\mathbf{U}^T\mathbf{Ux}=\mathbf{x}^T\mathbf{x}=\vert\mathbf{x}\Vert_{2}^2$$
因此, 正交变换 $\mathbf{x\to \mathbf{Ux}}$ 保持向量范数不变。
正交矩阵保持向量间夹角不变: 令 $\mathbf{x,y}$ 为单位向量, 则夹角 $\cos(\theta)=\mathbf{x}^{T}\mathbf{y}$， 在正交矩阵变换下, $\mathbf{x'=Ux,y'=Uy}$ 的夹角为 $\cos(\theta')=(\mathbf{x'})^{T}\mathbf{y}'$ . 由于
$$
(\mathbf{x'})^{T}\mathbf{y}'=(\mathbf{Ux}^T)(\mathbf{Uy})=\mathbf{x}^T\mathbf{U}^T\mathbf{Uy}=\mathbf{x}^T\mathbf{y}
$$
因此在正交变换下, 向量间夹角保持不变 $\cos(\theta)=\cos(\theta')$，反之亦真, 即保 持向量长度和夹角不变的变换必为正交变换。
## 秩一矩阵 (Rank-One Matrices)
如果矩阵 $\mathbf{A}\in\Bbb{R}^{m\times n}$ 能写成
$$
\mathbf{A}=\mathbf{u}^T\mathbf{v},u\in\Bbb{R}^m,v\in\Bbb{R}^n
$$
则称 $\mathbf{A}$ 是秩一矩阵 (rank-one matrix).
秩一矩阵对应一种特殊的线性变换: 对于输入 $\mathbf{x}\in\Bbb{R}^n$ 有：
$$
\mathbf{Ax}=(\mathbf{u}^T\mathbf{v})\mathbf{x}=(\mathbf{v}^T\mathbf{x})\mathbf{u}
$$
对于任意输入 $\mathbf{x}$ ，秩一矩阵对应的线性映射输出 (映射的像) 都是沿着向量 $\mathbf{u}$ 。 第二个等号是由于 $\mathbf{v}^T\mathbf{x}$ 是数值。
## 矩阵的迹 (Trace of Matrices)
### Def
$n\times n$ 矩阵 $\mathbf{A}$ 的迹 $tr(\mathbf{A})$ 定义为矩阵对角元素之和：
$$
tr(\mathbf{A})=\sum_{i=1}^n{A_{ij}}
$$
### 性质
- 转置的迹：$tr(\mathbf{A})=tr(\mathbf{A})$
- 迹的可交换性: 对于任意两个矩阵 $\mathbf{A,B\in\Bbb{R}^{m\times n}}$ ，有
$$
tr(\mathbf{A}^T\mathbf{B})=tr(\mathbf{B}^T\mathbf{A})
$$
- 对于 $\mathbf{x,y}\in\Bbb{R}^n$ ，有 $\mathbf{x}^T\mathbf{y}=tr(\mathbf{x}^T\mathbf{y})=tr(\mathbf{y}^T\mathbf{x})$
- 对于 $\mathbf{x,y}\in\Bbb{R}^n$ 和对称矩阵 $\mathbf{A}\in\Bbb{R}^{m \times n}$ ，有
$$
\mathbf{x}^T\mathbf{Ay}=tr(\mathbf{x}^T\mathbf{Ay})=tr(\mathbf{y}^T\mathbf{xA})=tr(\mathbf{Ay}\mathbf{x}^T)
$$
## 矩阵的内积
Recall: 所有 ${m \times n}$ 矩阵构成的集合相对于矩阵加法和数值乘法构成线 性空间, 此空间线性同构于 $\Bbb{R}^{m \times n}$。
矩阵的内积: 对于给定的 ${m \times n}$ 矩阵 $\mathbf{A,B}\in\Bbb{R}^{m \times n}$ , 定义矩阵内积为
$$
\langle\mathbf{A,B}\rangle=tr(\mathbf{A}^T\mathbf{B})=\sum^{i=1}_{m}\sum^{j=1}_{m}\mathbf{A}_{ij}\mathbf{B}_{ij}
$$
即，两个矩阵中对应元素乘积的和。
这相当于将矩阵展开成 ${m \times n}$ 向量，并计算对应的向量内积。
### 矩阵范数
回顾范数定义：如果映射 $\Vert\cdot\Vert:\Bbb{R}^{m \times n}\to\Bbb{R}$ 满足如下条件:
- 正定性：对于 $\forall \mathbf{A}\in \Bbb{R}^{m \times n}$ 都有 $\Vert\mathbf{x}\Vert\geq 0$ ，并且 $\Vert\mathbf{A}\Vert\iff\mathbf{A}=0$  
- 齐次性：对于 $\forall \mathbf{A}\in \Bbb{R}^{m \times n}$ 及 $\forall\alpha\in\Bbb{R}$ ，都有 $\Vert\alpha\mathbf{A}\Vert=\vert\alpha\vert\Vert\mathbf{A}\Vert$ 
- 三角不等式：$\Vert\mathbf{A+B}\Vert\leq\Vert\mathbf{A}\Vert+\Vert\mathbf{B}\Vert$  
则称 $\Vert\cdot\Vert$ 为 $\Bbb{R}^{m \times n}$ 上的矩阵范数。
- 对于任意给定的向量范数, 都可以诱导出一个矩阵范数, 称为谱范数：
$$
\Vert\mathbf{A}\Vert=\max_{x\not=0}\frac{\Vert\mathbf{Ax}\Vert}{\Vert\mathbf{x}\Vert}=\max_{\Vert\mathbf{x}\Vert=1}\Vert\mathbf{Ax}\Vert
$$
- Frobenius 范数将矩阵 $\mathbf{A}\in \Bbb{R}^{m \times n}$ 看做是 ${m \times n}$ 的向量,
$$
\Vert\mathbf{A}\Vert_{F}=\Bigg(  \sum^{i=1}_{m}\sum^{j=1}_{n}a_{ij}^2 \Bigg )^{\frac{1}{2}}
$$
# 线性方程组
## 矩阵的秩
${m \times n}$ 矩阵 $\mathbf{A}$ 对应一个线性映射 $\mathcal{A}$
$$
\begin{eqnarray}
\mathcal{A}:\Bbb{R}^n \to \Bbb{R}^m \\
\mathbf{x}\to \mathbf{Ax}
\end{eqnarray}
$$
集合 $Im(A)=\{\mathbf{Ax|x}\in\Bbb{R}^n\}$ 称为矩阵 $\mathbf{A}$ 的像（Range，Image）。
-  $Im(A)$ 表示对于任意输入向量 $\mathbf{x}\in\Bbb{R}^n$ ，其输出所能达到的范围。如 $\mathbf{y}\not\in Im(A)$ ，则不存在任何 $\mathbf{x}\Bbb{R}^n$ 使得 $\mathbf{y=Ax}$
-  $Im(A)$ 构成 $\Bbb{R}^m$ 的一个子空间。
子空间 $Im(A)$ 的维度称为矩阵 $\mathbf{A}$ 的秩（Rank）,记为 $rank(\mathbf{A})$ ，一般的， $rank(\mathbf{A})\leq\min(m,n)$
-  $rank(\mathbf{A})=\min(m,n)$ ，矩阵称为满秩的.。
> Fact: 对任意秩一矩阵 $\mathbf{A}\in\Bbb{R}^{m \times n}$ , 都存在 $\mathbf{x}\in\Bbb{R}^m,\mathbf{y}\in\Bbb{R}^n$ 使得
> $$
 \mathbf{A=x}\mathbf{y}^T
 $$

## 零空间
对于 $m \times n$ 矩阵 $\mathbf{A}$ ，集合
$$
\mathbf{N}(\mathbf{A}):=\{\mathbf{x}\in\Bbb{R}^n|\mathbf{Ax}=0\}
$$
称为矩阵 $\mathbf{A}$ 的零空间，或者核空间（Kernel）。 $\mathbf{N}(\mathbf{A})$ 也常写为 $ker(\mathbf{A})$ 。
- $\mathbf{N}(\mathbf{A})$ 构成 $\Bbb{R}^n$ 的一个子空间。
### 零化定理
令 $\mathcal{A}:\Bbb{R}^n\to\Bbb{R}^m$ ，则有
$$
dim(Im(\mathbf{A}))+dim(ker(\mathbf{A}))=n
$$
即，零子空间的维度和像空间的维度之和为矩阵的列，因此
$$
dim(ker(\mathbf{A}))=n-rank(\mathbf{A})
$$
## QR分解
对于 $m\times n$ 矩阵 $\mathbf{A}$ ，考虑求解线性方程组 $\mathbf{Ax}=\mathbf{y}$ 
$\mathbf{Basic\;Idea}$:
如果矩阵 $\mathbf{A}$ 是上三角矩阵 $\mathbf{A}_{ij}=0,i<j$ ，则上述线性方程组可以通过逐个变量带入进行快速求解，因此只需要将 $\mathbf{A}$ 变形为上三角矩阵。
$\mathbf{Theorem}$:
对于矩阵 $\mathbf{A}\in\Bbb{R}^{m \times n},m\geq  n$ ，假设其为列满秩 $rank(\mathbf{A})=n$ ，那么 $\mathbf{A}$ 有唯一的约化的 QR 分解 $\mathbf{A}=QR$ ，其中 $Q\in\Bbb{R}^{m\times n}$ 满足 $Q^TQ=I_n,R\in\Bbb{R}^{n\times n}$ 是对角元为正数 $r_{ij}>0$ 的上三角矩阵。对于线性方程组 $\mathbf{Ax}=\mathbf{y}$ ，两边同时乘以 $Q^T$ 可得
$$
Q^T\mathbf{Ax}=Q^TQR\mathbf{x}=R\mathbf{x}=Q^T\mathbf{y}
$$
因此，问题约简为上三角矩阵的线性方程组
- QR 分解可以通过Gram-Schmidt 正交化或 Householder 三角化完成
- QR 分解数值稳定性好，还可以用于估计矩阵的秩，和求解线性最小二乘问题。
- QR 分解的计算复杂度为 $O(4mn^2/3)$ 
# 对称矩阵的特征值分解
## 对称矩阵
一个矩阵 $\mathbf{A}\in\Bbb{R}^{m\times n}$ 称为对称矩阵，如果 $\mathbf{A}=\mathbf{A}^T$ ，即
$$
\mathbf{A}_{ij}=\mathbf{A}_{ji},1\leq i,j\leq n
$$
所有 $n\times n$ 对称矩阵所组成的集合称为 $\Bbb{S}^n$ 。
## 二次型
如下函数 $q:\Bbb{R}^n\to\Bbb{R}$ 称为二次型：
$$
q(\mathbf{x})=\sum_{i=1}^{n}\sum_{j=1}^{n}A_{ij}x_ix_j
$$
注意： $x_i^2$ 的系数为 $A_{ii}$ ，对于 $i\not=j$ 的项 $x_ix_j$ ，其系数为 $A_{ij}+A_{ji}$ .
令 $\mathbf{x}=(x_1,\cdots,x_n)^T\in\Bbb{R}^n$ 为 $n$ 维向量， $\mathbf{A}=(A_{ij})\in\Bbb{R}^{n\times n}$ 为 $n\times n$ 矩阵，则二次型可以表示为
$$
q(\mathbf{x})=\sum_{i=1}^{n}\sum_{j=1}^{n}A_{ij}x_ix_j=(x_1,\cdots,x_n)
\begin{pmatrix}
&A_{11}&\;\;\cdots\;\;&A_{1n}& \\
&\vdots& \;\; &\vdots& \\
&A_{n1}& \;\;\cdots\;\;&A_{nn}&
\end{pmatrix}
\begin{pmatrix}
x_1 \\
\vdots \\
x_n
\end{pmatrix}
=\mathbf{x}^T\mathbf{Ax}
$$

由于 $\mathbf{x}^T\mathbf{Ax}=tr(\mathbf{x}^T\mathbf{Ax})=tr(\mathbf{x}^T\mathbf{A}^T\mathbf{x})=\frac{1}{2}\mathbf{x}^T(\mathbf{A}+\mathbf{A}^T)\mathbf{x}$ ，我们可以假定二次型中矩阵 $\mathbf{A}$ 是对称矩阵。
因此，任何应该对称矩阵 $\mathbf{A}$ ，都对于一个二次型 $q(\mathbf{x})=\mathbf{x}^T\mathbf{Ax}$ 。
## 二次函数
函数 $q:\Bbb{R}^n \to \Bbb{R}$ 称为二次函数
$$
q(\mathbf{x})=\sum_{i=1}^{n}\sum_{j=1}^{n}A_{ij}x_ix_j+\sum_{i=1}^{n}b_{i}x_i+c
$$
其中 $A_{ij},b_i,c\in\Bbb{R},i,j\in\{1,\ldots,n \}$
一般的二次函数可以写成矩阵-向量形式：
$$
q(\mathbf{x})=\frac{1}{2}\mathbf{x}^T
\mathbf{Ax}+\mathbf{b}^T\mathbf{x}+c$$
二次函数中间 ，如果线性项和常数项为 0 ，即为二次型。
## 特殊对称矩阵-对角矩阵
### Def
只有对角元素为非零元素，所有非对角元素都是 0 的对称矩阵。
设 $\lambda\in\Bbb{R}^n$ ，以向量 $\lambda$ 为对角元的 $n\times n$ 对角矩阵一般记作：
$$
\mathbf{diag}(\lambda)=\mathbf{diag}(\lambda_1,\cdots,\lambda_n)
$$
### 对应的二次型
$$
q(\mathbf{x})=\mathbf{x}^T\mathbf{diag}(\lambda)\mathbf{x}=\sum_{i=1}^{n}\lambda_ix_i^2
$$
即对角矩阵对应的二次型不包括任何交叉项 $x_ix_j,i\not=j$ ，称为标准二次型。
## 特征值与特征向量
### Def
令 $\mathbf{A}$ 为 $n\times m$ 对称矩阵。如果 $\lambda\in\Bbb{R},\mathbf{u}\in\Bbb{R}^m,\mathbf{u}\not=0$ ，成立
$$
\mathbf{Au}=\lambda \mathbf{u}
$$
则称 $\lambda$ 为矩阵 $\mathbf{A}$ 的特征值，$\mathbf{u}$ 为对应特征值 $\lambda$ 的特征向量，如果 $\Vert\mathbf{u}\Vert_2=1$ ，则称为归一化特征向量，此时有
$$
\mathbf{u}^T\mathbf{Au}=\lambda \mathbf{u}^T\mathbf{u}=\lambda
$$
### 几何意义
矩阵 $\mathbf{A}$ 沿 $\mathbf{u}$ 方向的表现如同数值乘法。
矩阵 $\mathbf{A}$ 的特征值满足特征方程 $\det(\lambda I-\mathbf{A}=0)$ ，这是应该关于 $\lambda$ 的多项式。
> 对于对称矩阵，所有的特征值都是实数。

### 对称矩阵特征值分解
$$
\forall \mathbf{A}\in\Bbb{S}^n,\mathbf{A}=\mathbf{U\Lambda}\mathbf{U}^T=\sum_{i=1}^{n}\lambda_i\mathbf{u}_i\mathbf{u}_i^T,\Lambda=\mathbf{diag}(\lambda_1,\dots,\lambda_n)
$$
其中
 - 矩阵 $\mathbf{U}=[\mathbf{u_1},\dots,\mathbf{u}_n]$ 为正交矩阵，即 $\mathbf{U}^T\mathbf{U}=\mathbf{U}\mathbf{U}^T=1$ ，并且 $\Vert\mathbf{u}_i\Vert=1,\mathbf{u}_i^T\mathbf{u}_j=\delta_{ij}$
 - 对角矩阵 $\Lambda$ 的对角元为 $\mathbf{A}$ 的特征值。
 - 矩阵 $\mathbf{U}$ 的列 $\mathbf{u}_i$ 为 $\mathbf{A}$ 的对应 $\lambda$ 的特征向量 $\mathbf{Au}_i=\lambda_i\mathbf{u}_i$。
 $\lambda_i$ 为 $\mathbf{A}$ 的特征值，$\mathbf{u}_i$ 是其对应的特征向量。由于 $\mathbf{u}_i^T\mathbf{u}_j=\delta_{ij}$ 
$$
\mathbf{Au}_i=\sum_{i=1}^{n}\lambda_i\mathbf{u}_i\mathbf{u}_i^T\mathbf{u}_j=\sum_{i=1}^{n}\lambda_i\mathbf{u}_i(\mathbf{u}_i^T\mathbf{u}_j)=\lambda_i\mathbf{u}_i,j=1,\ldots,n
$$
## Eigen-decomposition of Symmetric Matrices
对称矩阵可以写成秩一矩阵的线性组合
$$
\mathbf{A}=\lambda_1\mathbf{u}_1\mathbf{u}_1^T+\ldots\lambda_n\mathbf{u}_n\mathbf{u}_n^T
$$
- 矩阵 $\mathbf{A}$ 的迹为： $tr(\mathbf{A})=\textstyle\sum_{i=1}^{n}\lambda_i$
- 特征矩阵 $\mathbf{U}$ 的每一列都是单位向量 $\Vert\mathbf{u}_i\Vert=1$
- $\mathbf{U}$ 的任意两列（任意两个特征向量）都是相互正交的：$\mathbf{u}_i^T\mathbf{u}_j=0$ 
- 矩阵 $\mathbf{A}$ 的秩 $rank(\mathbf{A})$ 等于其非零特征值的数目
$$
rank(\mathbf{A})=\sum_{i=1}^{n}\Bbb{1}(\lambda_i\not=0)
$$
- 如果矩阵 $\mathbf{A}$ 满秩, 则 $\mathbf{U}$ 的列向量构成线性空间 $\Bbb{R}^n$ 的一组正交基
### 对称矩阵的谱
对称矩阵 $\mathbf{A}$ 的特征分解
$$
\mathbf{A}=\lambda_1\mathbf{u}_1\mathbf{u}_1^T+\ldots\lambda_n\mathbf{u}_n\mathbf{u}_n^T,\lambda_1\geq \cdots \geq \lambda_n
$$
- 所有特征值的集合称为矩阵的谱 (spectral), 记为
$$
\sigma(\mathbf{A})=\{\lambda_1,\cdots,\lambda_n\}
$$
- 矩阵 $\mathbf{A}$ 的谱范数 (spectral norm)：
$$
\Vert\mathbf{A}\Vert=\max_{\sigma(\mathbf{A})}\vert\lambda_i\vert
$$
- 如果 $\mathbf{A}$ 为满秩矩阵
$$
\mathbf{A}^{-1}=\mathbf{U\Lambda}^{-1}\mathbf{U}^T
$$
## Rayleigh quotients
### Fact
设 $\mathbf{A}$ 为对称矩阵，$\lambda_{\min},\lambda_{\max}$ 分别为其最小和最大特征值，则成立
$$
\lambda_{\min}(\mathbf{A})=\min_{\mathbf{x}^T\mathbf{x}}{\mathbf{x}^T\mathbf{Ax}},\lambda_{\max}(\mathbf{A})=\max_{\mathbf{x}^T\mathbf{x}}{\mathbf{x}^T\mathbf{Ax}}
$$
上述结论表明，函数 $\mathbf{x}^T\mathbf{Ax}$ 在单位球 $\Vert\mathbf{x}\Vert=1$ 上的取值范围总是落在区间 $[\lambda_{min},\lambda_{max}]$ 的范围内。
等价于：
$$
\begin{eqnarray}
\lambda_{\min}(\mathbf{A})=\min_{\mathbf{x}^T\mathbf{x}=1}{\mathbf{x}^T\mathbf{Ax}}=\min_{\mathbf{x}\not=0}{\frac{{\mathbf{x}^T}\mathbf{Ax}}{{\mathbf{x}^T}\mathbf{x}}}, \\
\lambda_{\max}(\mathbf{A})=\max_{\mathbf{x}^T\mathbf{x}=1}{\mathbf{x}^T\mathbf{Ax}}=\max_{\mathbf{x}\not=0}{\frac{{\mathbf{x}^T}\mathbf{Ax}}{{\mathbf{x}^T}\mathbf{x}}}
\end{eqnarray}
$$
因此, 矩阵的特征值给出了 $\frac{{\mathbf{x}^T}\mathbf{Ax}}{{\mathbf{x}^T}\mathbf{x}}$ 的最大值和最小值, 此式称为瑞利商（Rayleigh quotient）。
- 对称矩阵的最大最小特征值分别可以看做一个二次型在球面 $\Vert\mathbf{x}\Vert=1$ 上的极大值极小值
## 矩阵范数
对于 $m\times n$ 矩阵 $\mathbf{A}$ ，定义
$$
\Vert\mathbf{A}\Vert:=\max_{\Vert\mathbf{x}\Vert_2=1}{\Vert\mathbf{Ax}\Vert_2}
$$
验证: 此定义构成 $m \times n$ 矩阵构成的线性空间 $\Bbb{R}^{m \times n}$ 上的一个范数. 这称为矩阵范数。
