# 向量微分与求导
## 可微函数
函数 $f:\Bbb{R}\to\Bbb{R}$ 在点 $x$ 处的导数定义为:
$$
f'(x)=\lim\limits_{t\to0}\dfrac{f(x+t)-f(x)}{t}\tag{1}
$$
函数 $f:\Bbb{R}\to\Bbb{R}$ 称 $f$ 在点 $x\in\Omega$ 处是可微的，如果存在向量 $\mathbf{g}$ 使得：
$$
\lim\limits_{\mathbf{p}\to0}\dfrac{f(\mathbf{x}+\mathbf{p})-f(\mathbf{x})-\langle\mathbf{g},\mathbf{p}\rangle}{\lVert\mathbf{p}\rVert}=0.
$$
其意义是，当 $\mathbf{p}\to{0}，f(x+\mathbf{p})-f(x)$ 与线性函数 $\langle\mathbf{g},\mathbf{p}\rangle=\mathbf{g}^T\mathbf{p}$ 的差，是 $\Vert\mathbf{p}\Vert$ 的高阶无穷小。
$$
f(\mathbf{y})=f(\mathbf{x})+\langle\mathbf{g},\mathbf{y}-\mathbf{x}\rangle+R(\|\mathbf{y}-{\mathbf{x}}\|),\quad\lim\limits_{\mathbf{y}\to\mathbf{x}}\dfrac{R(\|{\mathbf{y}}-\mathbf{x}\|)}{\|\mathbf{y-x}\|}=0.
$$
## 梯度
如果 $f$ 在点 $x\in\Omega$ 处是可微的, 则函数 $f(x)$ 连续, 并且
$$
f(\mathbf{y})=f(\mathbf{x})+\langle\nabla f(\mathbf{x}),~\mathbf{y}-\mathbf{x}\rangle+o(\|\mathbf{y} -\mathbf{x}\|)
$$
向量 $\nabla f(x)$ 为梯度
$$
\nabla f(\mathbf{x})=\dfrac{\partial f}{\partial x}=\left(\dfrac{\partial f}{{\partial x_1}},\cdots,\dfrac{{\partial f}}{{\partial x_n}}\right)^T\quad
$$
写成增量形式
$$
f(\mathbf x+\Delta\mathbf x)=f(\mathbf x)+\langle\nabla f(\mathbf x),\Delta\mathbf x\rangle+o(\|\Delta\mathbf x\|)
$$
## 微分
定义线性映射: $\text{d}f:\Bbb{R}\to\Bbb{R},\text{d}f(\Delta x)=\langle\nabla f(x),\Delta x\rangle$ 为函数的线性部分, 记为 $\text{d}f=\nabla f(x)\text{d}x$ ，称为 $x\in\Bbb{R}^n$ 处的微分
$$
\text{d}f=\nabla f(\mathbf{x})^T\text{d}\mathbf{x}=\langle\nabla f({\mathbf{x}}),\text{d}{\mathbf{x}}\rangle=\dfrac{\partial f}{\partial x_1}\text{d}x_1+\cdots+\dfrac{{\partial f}}{{\partial x_n}}\text{d}x_n\quad
$$
其中 $\text{d}x=(\text{d}x_{1},\ldots,\text{d}x_{n})^T$，函数 $f$ 在点 $x_{0}$ 的一次近似为
$$
f(\mathbf x_0)+\langle\nabla f(\mathbf x_0),\mathbf x-\mathbf x_0\rangle.
$$
## 方向导数
函数 $\text{d}f:\Bbb{R}\to\Bbb{R}$ 在点 $\mathbf{x}$ 处沿着方向 $\mathbf{d}\in\Bbb{R^n}$ 的方向导数定义为
$$
\dfrac{\partial f}{\partial\mathbf{d}}=\lim\limits_{t\to0}\dfrac{f(\mathbf{x}+t\mathbf{d})-f(\mathbf{x})}{t}
$$
- 对于沿坐标轴向向量 $e_{i}$ ，方向导数即为偏导数, 常记为 $\dfrac{\partial f}{\partial{x_{i}}}$
偏导数组成的向量称为梯度 $$\nabla f(\mathbf{x})=\left(\dfrac{\partial f}{{\partial x_1}},\cdots,\dfrac{{\partial f}}{{\partial x_n}}\right)^T\quad$$
代入公式 $(1)$ ，$f(\mathbf{x}+t\mathbf{d})=f(\mathbf{x})+t\langle\nabla f(\mathbf{x}),\mathbf{d}\rangle+o(\|\mathbf{d}\|)$，可得
$$
\dfrac{\partial f}{\partial\mathbf{d}}=\lim\limits_{t\to0}\dfrac{1}{t}\left(f(\mathbf{x}+t\mathbf{d})-f(\mathbf{x})\right)=\langle\nabla f(\mathbf{x}),\mathbf{d}\rangle=\nabla f(\mathbb{x})^T\mathrm{d}.
$$
## 向量微分与求导
### Def
对于函数  $\text{d}f:\Bbb{R}\to\Bbb{R}$，其微分可以写成
$$
\text{d}f=\sum_{i=1}^n\dfrac{\partial{f}}{\partial{x_i}}\text{d}x_i=\left(\dfrac{\partial f}{\partial\text{x}}\right)^T\text{d}\textbf{x}=\text{tr}\left(\left(\dfrac{\partial{f}}{{\partial\text{x}}}\right)^T\text{dx}\right)\quad
$$
其中$\text{d}x=(\text{d}x_{1},\ldots,\text{d}x_{n})^T$ 
> 如果微分能够写成 $\text{d}f=\text{tr}(\square^T\text{dx})$ 那么 $\square$ 就是函数对变量的梯度 $\square = \nabla f(\mathbf{x}) = \frac{\partial f}{\partial\mathbf{x}}$ 。

### 向量微分的运算法则
$$
\text{tr}A=\sum_{i=1}^n A_{ii}
$$
向量微分的运算法则：
- $\text{d tr}(X)=\text{tr}(\text{d}X),\quad\text{d}(X^T)=(\text{d}X)^T$
- $\text{d}(XY)=(\text{d}X)Y+X(\text{d}Y)$
### 迹运算规则
迹的运算规则：
- $\mathbf{x}^T\mathbf{y}=\mathbf{tr}(\mathbf x^T\mathbf{y})=\mathsf{tr}(\mathbf x\mathbf y^T)$
- $\mathbf{x}^T{\mathbf A}\mathbf y=\mathsf{tr}({\mathbf x}^T\mathbf{A}\mathbf y)=\mathop{\mathrm{tr}}(\mathbf y\mathbf x^T{\mathbf A})=\mathop{\mathbf{tr}}(\mathbf{A}\mathbf{y}\mathbf x^T)$
### 向量求导
**Example**：$\dfrac{\partial\mathbf{x}^T\mathbf{y}}{\partial\mathbf{x}}$
$$
\text{d}f=\text{d}\mathbf{x}^{T}\mathbf{y}=\text{tr}(\text{dx}^T\mathbf{y})=\text{tr}(\mathbf{y}^T\text{dx})
$$
即 $\text{d}f=\text{tr}(\mathbf{y}^T\text{d}\mathbf{x})，\frac{\partial\mathbf{x}^T\mathbf{y}}{\partial\mathbf{x}}=\mathbf{y}.$
# 映射微分与链式法则
## 映射的微分
### Def
映射 $F\colon\mathbb{R}^n\to\mathbb{R}^m,\mathbf{x}\mapsto F(\mathbf{x})=\left(f_1(\mathbf{x}),\cdots,f_m(\mathbf{x})\right)^T$。称 $F$ 在点 $x\in\Omega$ 处是可微的，如果存在矩阵 $\mathbf{A}$ 使得
$$
F({\mathbf{y}})=F({\mathbf{x}})+{\mathbf{A}}({\mathbf{y}}-{\mathbf{x}})+{{\mathbf{R}}}{(\mathbf{y}-{\mathbf{x}})},\lim\limits_{{\mathbf{y}}\to{\mathbf{x}}}\dfrac{\|{\mathbf{R}}(\mathbf{y}-\mathbf{x})\|}{\|{\mathbf{y}}-\mathbf{x}\|}=0.\quad
$$
即当 $y\to x,F(\mathbf{y})-F(\mathbf{x})-\mathbf{A}(\mathbf{y-x})$ 是 $\Vert\mathbf{y-x}\vert$ 的高阶无穷小。
### 性质
映射 $F\colon\mathbb{R}^n\to\mathbb{R}^m,\mathbf{x}\mapsto F(\mathbf{x})=\left(f_1(\mathbf{x}),\cdots,f_m(\mathbf{x})\right)^T$ 在点 $x\in\Omega$ 处是可微的，则函数 $f(x)$ 连续，$\dfrac{\partial f_i}{\partial x_j}$ 存在，且
$$
\mathbf{A}=\bigg(\dfrac{\partial f_i}{\partial x_j}\bigg)\bigg|_{\mathbf{x}_0}=\begin{pmatrix}\frac{\partial f_1}{\partial x_1}&\cdots&\frac{\partial f_{1}}{\partial x_n}\\ \vdots&&\vdots\\ \frac{\partial f_{m}}{\partial x_1}&\dots&\frac{\partial f_m}{\partial x_n}\end{pmatrix}.
$$
规定：
$$
\nabla_{\mathbf{x}}F(\mathbf{x})=\dfrac{\partial F}{\partial\mathbf{x}}=\begin{pmatrix}\frac{\partial f_1}{\partial x_1}&\cdots&\frac{\partial f_n}{\partial x_1}\\ \vdots&&\ddots\\ \frac{\partial f_{1}}{\partial x_n}&\cdots&&\frac{\partial f_m}{\partial x_n}\end{pmatrix},\quad\mathbf{A}=\begin{pmatrix}{\partial f_1}&\dots&\frac{\partial f_{1}}{\partial x_n}\\ \vdots&&\vdots\\\frac{\partial f_{m_1}}{\partial x_1}&\dots&&\frac{\partial f_{m}}{\partial\mathbf{x_n}}\end{pmatrix}
$$
即矩阵 $\mathbf{A}$ 为映射 $F$ 的 $Jacobi$ 矩阵的转置 $\mathbf{A}=\nabla F(\mathbf{x})^T=(\frac{\partial F}{\partial\mathbf{x}})^T$
>$\mathbf{A}$ 为 $m\times n$ 矩阵， $\frac{\partial F}{\partial\mathbf{x}}$ 为 $n\times m$ 矩阵

因此, 写成增量形式, 有
$$
F(\mathbf x+\Delta\mathbf x)=F(\mathbf x)+\mathbf A\Delta\mathbf x=F(\mathbf x)+\nabla F(\mathbf x)^T\Delta\mathbf x
$$
$\text{d}F_\mathbf{x}(\Delta\mathbf{x})=\mathbf{A}\Delta\mathbf{x}=\nabla F(\mathbf{x})^T\Delta\mathbf{x}$ 为函数的线性部分，称为映射 $F$ 在 $\mathbf{x}\in\Bbb{R}^n$ 处的微分，记为 $\text{d}F=\nabla F(\mathbf{x})^T\text{dx}$
### 微分的链式法则
给定映射 $g\colon\mathbb{R}^n\to\mathbb{R}^m\not\forall x f\colon\mathbb R^m\to\mathbb{R},\not\to\mathbb{u}=g(\mathbf x)$ ，令 $\mathbf{u}=g(\mathbf{x})$ ，考虑复合函数 $f\circ g:\mathbb{R}^n\to\mathbb{R}$ ，其微分为
$$
\text{d}(f\circ g)=\left(\dfrac{\partial f}{\partial\mathbf{u}}\right)^T\text{d}\mathbf{u}=\left(\dfrac{{\partial f}}{{\partial}\mathbf{u}}\right){}^T\left(\dfrac{\partial\mathbf{u}}{{\partial\mathbf{x}}}\right)^T\text{dx}=\left(\dfrac{\partial{\mathbf{u}}}{{\partial}\mathbf{x}}\dfrac{\partial f}{{\partial}{\mathbf{u}}}\right){}^T\text{dx}
$$
因此：
$$
\nabla f\circ g(\mathbf{x})=\dfrac{\partial f\circ g(\mathbf x)}{\partial\mathbf x}=\dfrac{\partial\mathbf u}{\partial\mathbf x}\dfrac{\partial f}{\partial\mathbf u}
$$
## 向量求导: 线性回归模型
数据集 $\{(\mathbf{x}_i,y_i)\},\mathbf{x}_i\in\mathbb{R}^d_n,\mathbf{X}=(\mathbf{x}_1,\cdots,x_N)\in\mathbb{R}^{d\times N}_n\mathbf{y}=(y_1,\cdots,y_N)^T$ 
线性回归模型 $g(\mathbf{w})=\mathbf{x}_{i}^T\mathbf{w}$ ，目标函数 $\ell(\mathbf{w})$ ，计算 $\frac{\partial\ell}{\partial\mathbf{w}}$ 
$$
\ell(\mathbf w)=\sum_i(\mathbf x_i^T\mathbf w-y_i)^2=\|\mathbf X^T\mathbf w-\mathbf y\|^2
$$
$\mathbf{Ans:}$
由内积与范数关系，有
$$
\ell(\mathbf{w})=\|\mathbf{X}^{T}\mathbf{w}-\mathbf{y}\|^{2}=\left(\mathbf{X}^{T}{\mathbf{w}}-\mathbf{y}\right)^{T}(\mathbf{X}^T\mathbf{w}+\mathbf{y})
$$
计算其微分，有
$$
\begin{eqnarray}
\quad\text{d}\ell&=&(\mathbf{X}^{T}\mathbf{d}\mathbf{w})^T(\mathbf{X}^T\mathbf{w}-\mathbf{y})+(\mathbf{X}^T{\mathbf{w}}-\mathbf{y})^T({\mathbf{X}}^T{\mathbf{d}}\mathbf{w})\quad
\\
&=&2\mathbf{tr}(\mathbf{X}^{T}\mathbf{w}-\mathbf{y})^{T}(\mathbf{X}^T\mathbf{d}\mathbf{w}) \\ &=&2\mathbf{tr}\left((\mathbf{X}^{T}{\mathbf{w}}-{\mathbf{y}})^{T}{\mathbf{X}^T}{\mathbf{d}}\mathbf{w}\right)
\end{eqnarray}
$$
因此：
$$
\dfrac{\partial\ell}{\partial\mathbf{w}}=2\mathbf{X}(\mathbf{X}^T\mathbf{w}-\mathbf{y})
$$
# 矩阵函数求导
## 矩阵变量函数的导数: 定义法
矩阵导数. 对于以 $m \times n$ 矩阵 $\mathbf{X}$ 为自变量的函数 $f(\mathbf{X})$ ，若存在矩阵 $\mathbf{G}\in\Bbb{R}^{m \times n}$ 满足
$$
\lim\limits_{\mathbf{V}\to0}\dfrac{f(\mathbf{X}+\mathbf{V})-f(\mathbf{X})-\langle\mathbf{G},\mathbf{V}\rangle}{\|\mathbf{V}\|}=0
$$
称矩阵变量函数 $f$ 在 $\mathbf{X}$ 处可微。矩阵 $\mathbf{G}$ 称为函数 $f$ 对矩阵 $\mathbf{X}$ 的梯度/导数。矩阵导数可以表示为
$$
\nabla f(\mathbf{X})=\begin{pmatrix}\frac{\partial f}{\partial x_{11}}&\cdots&\frac{\partial f}{\partial{x}_{1n}}\\ \vdots&&\vdots\\ \frac{\partial f}{\partial{{x}_{n1}}}&\cdots&\frac{{\partial f}}{\partial{x}_{nn}}\end{pmatrix}
$$
写成增量形式
$$
f(\mathbf{X}+\mathbf{V})=f(\mathbf{X})+\langle\nabla f(\mathbf{X}),\mathbf{V}\rangle+o(\|\mathbf{V}\|)
$$
## 矩阵变量函数的导数: Gâteaux 梯度法
设 $f(\mathbf{X})$ 是矩阵函数，如果对于任意方向 $\mathbf{V}\in\Bbb{R}^{m \times n}$ ，存在矩阵 $\mathbf{G}\in\Bbb{R}^{m \times n}$ 满足
$$
\lim\limits_{t\to0}\dfrac{f(\mathbf{X}+t\mathbf{V})-f(\mathbf{X})-t\langle\mathbf{G},\mathbf{V}\rangle}{t}=0
$$
结论: 如果 $f$ 可微函数, $f$ 也是 Gâteaux 可微的, 且两种梯度相等． 此结论给出了一种计算矩阵导数的方法, 即如果能写成
$$
f(\mathbf{X}+t\mathbf{V})=f(\mathbf{X})+t\langle\mathbf{G},\mathbf{V}\rangle+O(t^2)
$$
则矩阵 G 就是函数的矩阵导数/梯度。
## 矩阵变量函数的微分: 迹法
增量形式： $f(\mathbf{X}+\mathbf{V})=f(\mathbf{X})+\langle\nabla f(\mathbf{X}),\mathbf{V}\rangle+o(\|\mathbf{V}\|)$，其中，
$$
\nabla f(\mathbf{X})=\dfrac{\partial f}{\partial\mathbf{X}}=\begin{pmatrix}\frac{\partial f}{\partial x_{11}}&\cdots&\frac{\partial f}{\partial{x}_{1n}}\\ \vdots&&\vdots\\ \frac{\partial f}{\partial{{x}_{n1}}}&\cdots&\frac{{\partial f}}{\partial{x}_{nn}}\end{pmatrix}
$$
矩阵函数微分: 对于矩阵变量函数 $F\colon\mathbb{R}^{m \times n}\to\mathbb{R}$，其微分为
$$
\text{d}f=\sum_{i=1}^m\sum_{j=1}^n\dfrac{\partial f}{\partial X_{ij}}\text{d}X_{ij}=\text{tr}\left(\left(\frac{\partial f}{\partial\mathbf{X}}\right)^Td\mathbf{X}\right)\quad
$$
> $\text{tr}(A^TB)=\sum_{i=1}^m\sum_{j=1}^n A_{ij}B_{ij}$
>如果矩阵变量函数微分能够写成 $\text{d}f=\text{tr}(\square^T\text{d}\mathbf{x})$ 那么 $\square$ 就是函数对变量的梯度 $\square = \nabla f(\mathbf{X}) = \frac{\partial f}{\partial\mathbf{X}}$ 。

## 矩阵微分的运算法则
- $\text{d tr}(X)=\text{tr}(dX)$，因此 $\frac{\partial\text{tr}X}{\partial X}=I$ 
- $\text{d}(XY)=(\text{d}X)Y+X(\text{d}Y)$
- $\text{d}(X^T)=(\text{d}X)^T$ 
# 微分中值定理
## 二阶近似
令 $f:\Bbb{R}^n\to\Bbb{R}$ 是二阶连续可微函数, 对于 $\mathbf{d}\in\Bbb{R}^n$ , Taylor 展开式为
$$
f(\mathbf{x}+\mathbf{d})=f(\mathbf{x})+\nabla f(\mathbf{x})^T\mathbf{d}+\dfrac{1}{2}\mathbf{d}^T\nabla^2f(\mathbf{x})\mathbf{d}+o(\|\mathbf{d}\|^2)
$$
其中 $\nabla^2f(\mathbf{x})$ 为 Hessian 矩阵
**First order**: 函数 $f$ 在点 $x_{0}$ 的一阶函数近似
$$
f(\mathbf x_0)+\langle\nabla f(\mathbf x_0),\mathbf x-\mathbf x_0\rangle
$$
**Second order**: 函数 $f$ 在点 $x_{0}$ 周围的二阶近似
$$
f(\mathbf x_0)+\langle\nabla f(\mathbf x_0),\mathbf{x}-\mathbf x_0\rangle+\dfrac{1}{2}(\mathbf x-\mathbf x_0)^T\nabla^2f(\mathbf x_0)(\mathbf x-\mathbf{x}_0)
$$
## 转换函数与方向导数
### Theorem
$f:\Bbb{R}^n\to\Bbb{R}$ 是连续可微函数，令 $g(t)=f(\mathbf{x}+t\mathbf{y})$ ，有
$$
g'(t)=\dfrac{\partial f(\mathbf{x}+t\mathbf{y})}{\partial t}=\nabla f(\mathbf{x} + t\mathbf{y})^T\mathbf{y}
$$
### Proof
直接对 $g(t)=f(\mathbf{x}+t\mathbf{y})$ 求导：
$$
\begin{aligned}\lim\limits_{\Delta t\rightarrow0}\frac{g(t+\Delta t)-g(t)}{\Delta t}&=\lim\limits_{{\Delta t\rightarrow0}}\frac{f(\mathbf{x}+(t+\Delta t)\mathbf{y})-f(\mathbf{x}+t\mathbf{y})}{\Delta t}\\ &=\lim\limits_{i t\rightarrow0}\frac{{f(x+t\mathbf{y}+\Delta t\cdot\mathbf{y})-f({\mathbf{x}}+t\mathbf{y}})}{\Delta t}\end{aligned}
$$
根据方向导数定义, 此即 $f$ 在 $(\mathbf{x}+t\mathbf{y})$ 处沿着 $\mathbf{y}$ 的方向导数
$$
g'(t)=\nabla f(\mathbf{x}+t\mathbf{y})^T\mathbf{y}
$$
如果令 $g(t)=f(\textbf{x}+t(\textbf{y}-\textbf{x}))$ ，有
$$
g'(t)=\dfrac{\partial f(\mathbf{x}+t(\mathbf{y}-\mathbf{x}))}{\partial t}=\langle\nabla f({\mathbf{x}}+t({\mathbf{y}}-\mathbf{x})),~{\mathbf{y}}-{\mathbf{x}}\rangle
$$
## 转换函数与方向曲
### Theorem
$f:\Bbb{R}^n\to\Bbb{R}$ 是二阶连续可微函数，令 $g(t)=f(\mathbf{x}+t\mathbf{y})$ ，有
$$
g''(t)=\dfrac{\partial^2f(\mathbf{x}+t\mathbf{y})}{\partial t^2}=\mathbf{y}^T\nabla^2f(\mathbf x+t\mathbf{y})\mathbf{y}
$$
### Proof
由 $g'(t)=\nabla f(\mathbf{x} + t\mathbf{y})^T\mathbf{y}$ ，
$$
g''(t)=\mathbf{y}^T\nabla^2f(\mathbf{x}+t\mathbf{y})\mathbf{y}
$$
> $\frac{\partial^2f(\mathbf{x})}{\partial\mathbf{y}^2} = \mathbf{y}^T\nabla^2 f(\mathbf{x})\mathbf{y}$ 为 $f$ 沿着 $\mathbf{y}$ 方向的方向曲率。

如果令 $g(t)=f(\textbf{x}+t(\textbf{y}-\textbf{x}))$ ，有
$$
g''(t)=\dfrac{\partial^2f(\mathbf{x}+t(\mathbf{y}-\mathbf{x}))}{\partial t^2}=(\mathbf{y-x})^T\nabla^2f(\mathbf x+t(\mathbf{y-x}))(\mathbf{y-x})
$$
## 微积分基本定理
微积分基本定理: 函数 $f:[a,b]\to\Bbb{R}$ 是连续可微函数, 则
$$
\int_a^b f'(x)\text{d}x=f(b)-f(a)
$$
### Theorem 1
$f:\Bbb{R}^n\to\Bbb{R}$ 是连续可微函数，对于 $\mathbf{x},\mathbf{y}\in\Bbb{R}^n$ ，定义 $g(t)=f(\textbf{x}+t(\textbf{y}-\textbf{x})),g(0)=f(\mathbf{x}),g(1)=f(\mathbf{y})$ ，有
$$
\int_0^1g′'(t)\mathrm{d}t=f(\mathbf{y})-f(\mathbf{x})
$$
### Theorem 2
$f:\Bbb{R}^n\to\Bbb{R}$ 是二阶连续可微函数，对于 $\mathbf{x},\mathbf{y}\in\Bbb{R}^n$ ，定义 $g(t)=f(\textbf{x}+t(\textbf{y}-\textbf{x})),g(0)=f(\mathbf{x}),g(1)=f(\mathbf{y})$ ，有
$$
\int_0^1g''(t)\text{d}t=\langle\nabla f(\mathbf{y})-\nabla f({\mathbf{x}}),\mathbf{y}-{\mathbf{x}}\rangle
$$
**Proof**：
由于
$$
g'(t)=\langle\nabla f({\mathbf{x}}+t({\mathbf{y}}-\mathbf{x})),~{\mathbf{y}}-{\mathbf{x}}\rangle
$$
因此 $g'(1)=\langle\nabla f(\mathbf{y}),\mathbf{y}-\mathbf{x}\rangle,g'(0)=\langle{\nabla f}(\mathbf{x}),\mathbf{y}\mathbf{-x}\rangle$ 代入可得
$$
\int_0^1g''(t)\text{d}t=g'(1)-g'(0)=\langle\nabla f(\mathbf{y})-\nabla f({\mathbf{x}}),\mathbf{y}-{\mathbf{x}}\rangle
$$
## 微分中值定理
这是单变量 Lagrange 中值定理的自然推广.
### Theorem
$f:\Bbb{R}^n\to\Bbb{R}$ 是连续可微函数， $\mathbf{p}\in\Bbb{R}^n$ ，那么存在 $\xi\in(0,1)$ 使得
$$
f(\mathbf x+\mathbf p)-f(\mathbf x)=\langle\nabla f(\mathbf x+\xi\mathbf p),\mathbf p\rangle
$$
如果 $f$ 二阶连续可微
$$
f(\mathbf{x+p})-f(\mathbf{x})=\langle\nabla f(\mathbf{x}),\mathbf{p}\rangle+\dfrac{1}{2}\mathbf{p}^T\nabla^2f(\mathbf{x}+\xi\mathbf{p})\mathbf{p}
$$
证明: 令 $g(t)=f(\mathbf{x}+t\mathbf{y}),,g(0)=f(\mathbf{x})$ ，并且 $g'(t)=\nabla f(\mathbf{x}+t\mathbf{p})^T\mathbf{p}$。
## Taylor 展开式: Lagrange 余项与积分余项
### Theorem
$f:\Bbb{R}^n\to\Bbb{R}$ 是二阶连续可微函数，$\mathbf{p}\in\Bbb{R}^n$ ，那么存在 $t\in(0,1)$ 使得
$$
\begin{eqnarray}
&f&(\mathbf{x}+\mathbf{p})-f(\mathbf{x})=\langle\nabla f(\mathbf{x}),\mathbf{p}\rangle+\frac{1}{2}\mathbf{p}^T\nabla^2f(\mathbf{x+\xi}\mathbf{p})\mathbf{p} \\
&f&(\mathbf{x}+{\mathbf{p}})-f(\mathbf{x}\rangle=\langle{\nabla f}(\mathbf{x}),{\mathbf{p}}\rangle+\int_{0}^{1}(1-t)\mathbf{p}^T{\nabla^2}f(\mathbf{x+tp})\mathbf{p}\mathrm{d}t
\end{eqnarray}
$$
第一式, 构造 $g(t)=f(\mathbf{x}+t\mathbf{p})$，使用Taylor 展开式的Lagrange 余项可得，第二式等价于
$$
f(\mathbf{y})=f(\mathbf{x})+\nabla f(\mathbf{x})^{\top}(\mathbf{y}-\mathbf{x})+\int_{0}^{1}(1-t)\dfrac{\partial^{2}f(\mathbf{x}+t({\mathbf{y}}-\mathbf{x}))}{\partial t^{2}}\text{d}t
$$
## Taylor 展开式积分余项
### Theorem
$f:\Bbb{R}^n\to\Bbb{R}$ 是二阶连续可微函数，$\mathbf{p}\in\Bbb{R}^n$ ，则
$$
f(\mathbf{y})=f(\mathbf{x})+\nabla f(\mathbf{x})^{\top}(\mathbf{y}-\mathbf{x})+\int_{0}^{1}(1-t)\dfrac{\partial^{2}f(\mathbf{x}+t({\mathbf{y}}-\mathbf{x}))}{\partial t^{2}}\text{d}t
$$
### Proof
构造 $g(t)=f(\textbf{x}+t(\textbf{y}-\textbf{x})),g(0)=f(\mathbf{x}),g(1)=f(\mathbf{y})$ 
$$
g(1)=g(0)+g'(0)+\int_0^1g''(t)(1-t)\text{d}t
$$
此即为积分余项. 由此可得
$$
f(\mathbf{y})=f(\mathbf{x})+\nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})+\int_0^1g''(t)(1-t)\text{d}t
$$
代入 $g''(t)$ 的表达式，即得。( $g(t)=f(\mathbf{x}+t\mathbf{p}$ 可得另一个形式 )
# 切空间与等高线
## 曲线的切向量
连续映射 $\mathbf{x}:(a,b)\to\mathbb{R}^n,t\mapsto\mathbf{x}(t)=(x_1(t),\cdots,x_n(t))^T$ ，表示空间 $\Bbb{R}^n$      中的一条曲线。
向量 $\mathbf{x'}(t)=(x'_1(t),\cdots,x'_n(t))^T$ 称为曲线 $\mathbf{x}(t)$ 在点 $t$ 处的切向量(tangent vector), 其中
$$
x'_i(t)=\lim\limits_{\Delta t\to0}\dfrac{x_i(t+\Delta t)-x_i(t)}{\Delta t}
$$
如果 $\mathbf{x'}(t)$ 连续可微, 则称 $\mathbf{x}(t)$ 称为连续可微曲线。
## 曲面的切向量
曲面: $\Bbb{R}^3$ 空间中的曲面可以表示为 $X:\mathbb{R}^2\to\mathbb{R}^3,(u,v)\mapsto(x,y,z).$ 一般 可以表示为 $(u,v)$ 的参数函数
$$
F(u,v)=(x(u,v),y(u,v),z(u,v))
$$
曲面的集合表示为 $\{(x(u,v),y(u,v),z(u,v))\in\mathbb R^3|(u,v)\in\mathbb R^2\}$ 
曲面 $M$ 上的一条曲线 $c:\Bbb{R}\to M$ 可以表示为
$$
\begin{pmatrix}\tilde{x}(t)\\ \tilde{y}(t)\\\tilde{z}(t)\end{pmatrix}=\begin{pmatrix}x(u(t),v(t))\\ y(u(t),{v}(t))\\ z(u(t),{v(t))}\end{pmatrix}
$$
即曲线在每一个维度分量上的值可以看成是 $t$ 的复合函数. 假设 $c(0)=\mathbf{x}\in M$ , 则 $c(0)$ 为曲面 $M$ 点 $\mathbf{x}$ 处的切向量 $(\tilde{x}'(t),\tilde{y}'(t), \tilde{z}'(t))^T$
所有过点 $\mathbf{x}\in M$ 的可微曲线的切线, 都是点 $\mathbf{x}$ 处的切向量. 所有过 $\mathbf{x}$ 的 切向量构成的集合记为 $T_{\mathbf{x}}M$。
## 切空间
曲面 $M$ 上点 $\mathbf{x}$ 的所有的切向量构成一个线性空间 $T_{\mathbf{x}}M\simeq\Bbb{R}^2$  , 称为 $\mathbf{x}$ 点处的切空间。
- 曲面 $M$ 的两个坐标向量为
$$
\dfrac{\partial F}{\partial u}=\left({\dfrac{\partial x}{\partial u}},{\dfrac{\partial y}{\partial u}},\dfrac{\partial z}{\partial u}\right)^T,\dfrac{\partial F}{\partial v}=\left({\dfrac{\partial x}{\partial v}},{\dfrac{\partial y}{\partial v}},\dfrac{\partial z}{\partial v}\right)^T
$$
- 任何切向量都可以写成其线性组合: $\textbf{d}=a \frac{\partial F}{\partial u}+b\frac{\partial F}{\partial v}$
### Def
对于高维曲面 $\mathbb{R}^n\to\mathbb{R}^{n+1},\mathbf{x}\mapsto F=(F_1(\mathbf{x}),\cdots,F_{n+1}(\mathbf{x}))$ 过点 $\mathbf{x}$ 处所有曲线的切线构成一个线性空间 $T_{\mathbf{x}}M$ , 称为切空间。
切空间的基：向量组 $\left\{\frac{\partial F}{\partial x_1},\cdots,\frac{\partial F}{\partial{x_i}},\cdots,\frac{{\partial F}}{{\partial x_n}}\right\}$ 其中：
$$
\frac{\partial F}{\partial x_i}=\begin{pmatrix}\frac{\partial F_1}{\partial x_i}\\ \vdots\\ \frac{\partial F_n}{\partial x_i}\end{pmatrix}
$$
构成切空间 $T_{\mathbf{x}}M$ 的一组基，任何切向量都可以表示为 $\mathbf{d}=\sum_i a_i\frac{\partial}{\partial x_i}$ 。
所以切空间维度与曲面维度相同, 均为自由变量的个数 $n$ 。
## 水平集（Level-Set）
给定函数 $f\colon\mathbb{R}^n\to\mathbb{R}$ ，集合 $S=\{\mathbf{x}\in\mathbb{R}^n|f(\mathbf{x})=c\}$ 称为水平集。
如果 $f\colon\mathbb{R}^n\to\mathbb{R}$ 连续可微，并且 $f(\mathbf{x})$ 非常数，则 $S=\{\mathbf{x}\in\mathbb{R}^n|f(\mathbf{x})=c\}$ 为空集，或者 $n-1$ 为连续可微曲面，称为登高面。
## 等高面的切线
设函数 $f(\mathbf{x})$ 连续可微，$S=\{\mathbf{x}\in\mathbb{R}^n|f(\mathbf{x})=c\}$ 为等高面，过 $S$ 上一点 $\mathbf{x}_{0}\in S$ 的任意切向量 $\mathbf{v}$ 与梯度 $\nabla f(\mathbf{x}_{0})$ 相互正交$\langle\nabla f(\textbf{x}_0),\textbf{v}\rangle=0$ 。
**Proof：**
由微分方程解的存在性, 设 $\mathbf{x}_{t}$ 为等高面 $S$ 上的过 $\mathbf{x}_{0}$ 且以 $\mathbf{v}$ 为切向量的连续可微曲线，并且 $\mathbf{x}(t)|_{t=0}=\mathbf{x}_0$ ，则有 $f(\mathbf{x}(t))=c$ 。 两边对 $t$ 求导, 有 (使用了链式法则)：
$$
\left.\dfrac{\text{d}f}{\text{d}t}\right|_{t=0}=\left(\dfrac{\partial f}{\partial\text{x}}\right)^T\dfrac{\text{d}\textbf{x}(t)}{\text{d}t}{\Big|}_{t=0}=\nabla f(\textbf{x})^T\dfrac{{\text{d}}\textbf{x}(t){\text{d}t}}{\text{d}t}\Big|_{t=0}={\langle\nabla f}(\textbf{x}_0),\textbf{v}\rangle=0
$$
即, 梯度 $\nabla f(\mathbf{x}_{0})$ 与等高面 $S$ 上的 $\mathbf{x}_{0}$ 点处的切向量 $\mathbf{v}$ 相互正交。由于 $\mathbf{v}$ 的任意性，梯度 $\nabla f(\mathbf{x}_{0})$ 与等高面 $S$ 上 $x_{0}$ 的切平面正交，即梯度 $\nabla f(\mathbf{x}_{0})$ 是登高面 $S$ 上 $\mathbf{x}_{0}$ 点处的法向量。梯度 $\nabla f(\mathbf{x}_{0})$ 与等高面 $S$ 上的切方向正交。