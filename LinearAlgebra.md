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
$$\mathbf{S}=span(\mathbf{x}_1,\ldots,\mathbf{x}m):=\{
\sum_{i=1}^{n}\lambda_i\mathbf{v}_i|\lambda_i\in \Bbb{R}\}$$
## 向量空间的自然基
### Def
一组向量 $\mathbf{v_1,v_2,\cdots,v_k}$ 称为向量空间 $V$ 的一组基，如果它是最大的线性无关集合，即对于任意的 $x\in V$ ，都存在系数 $\lambda_1,\ldots,\lambda_n$，使得 $\mathbf{x}$ 可以由此向量组线性表示
$$\mathbf{x}=\sum_{i=1}^{n}\lambda_i\mathbf{v}_i$$
### 自然基
由向量 $\mathbf{e}_i$ 构成的向量组，其中 $\mathbf{e}_i$ 仅第 $i$ 分量为1，其余均为0：
$$\mathbf{e}_i=(0,\ldots,1,\ldots,0)$$
例如：对于 $\Bbb{R}^3$ ,有
$$\mathbf{e_1}=\begin{pmatrix}1 \\ 0 \\ 0 \end{pmatrix},\mathbf{e_2}=\begin{pmatrix}0 \\ 1 \\ 0 \end{pmatrix},\mathbf{e_3}=\begin{pmatrix}1 \\ 0 \\ 1 \end{pmatrix}$$
### 子空间的基
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
### 向量范数（Norm）
向量范数是对向量“长度”的刻画。
### Def
令 $V$ 为线性空间/向量空间，$\Vert\cdot\Vert:V\rightarrow\Bbb{R}$ 为一个映射，如果映射  $\Vert\cdot\Vert$ 满足以下性质，则称其为一个范数。
- 正定性：对于任意 $\mathbf{x}\in V$ 都有 $\Vert\mathbf{x}\Vert\geq 0$ ，并且 $\Vert\mathbf{x}\Vert\iff\mathbf{x}=0$  
- 齐次性：对于任意 $\mathbf{x}\in V$ 及 $\alpha\in\Bbb{R}$ ，都有 $\Vert\alpha\mathbf{x}\Vert=\vert\alpha\vert\Vert\mathbf{x}\Vert$ 
- 三角不等式：$\Vert\mathbf{x+y}\Vert\leq\Vert\mathbf{x}\Vert+\Vert\mathbf{x}\Vert$  

范数给出了度量向量“方式”的一种方式
任何内积都能引诱一种对应的范数，对于内积空间 $(\Bbb{V},\langle\cdot,\cdot\rangle)$ ，可以相应定义
$$\Vert\mathbf{x}\Vert=\sqrt{\langle\mathbf{x},\mathbf{x}\rangle}$$
