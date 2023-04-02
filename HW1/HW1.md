备份及代码在[GitHub链接]([OptiMark/HW1 at main · ashengstd/OptiMark (github.com)](https://github.com/ashengstd/OptiMark/tree/main/HW1))
# 1.1.1.1
## 1
### (a)
$$
\begin{eqnarray}
f(\mathbf{x}) &=& \|A\mathbf{x-y}\|^2 \\
&=&\sum^{n}_{i=1}\sum^{n}_{j=1}(A_{ij}x_{j}y_{j})^2 \\
&=&\sum^{n}_{i=1}\sum^{n}_{j=1}A_{ij}^2x_{k}-2A_{ij}y_{j}x_{j}+y_{j}^2 \\
\therefore\frac{\partial f}{\partial x_{i}}&=&\sum^{n}_{k=1}2A_{ik}x_{k}^2-2A_{ik}y_{k} \\
\end{eqnarray}
$$
### (b)
$$
\begin{eqnarray}
f(\mathbf{x}) &=& \|A\mathbf{x-y}\|^2=(A\mathbf{x-y})^T(A\mathbf{x-y}) \\
\text{d}f&=&(A\text{d}\mathbf{x})^T(A\mathbf{x-y})+(A\mathbf{x-y})A\text{d}\mathbf{x} \\
&=&2tr(A\mathbf{x-y})^TA\text{d}\mathbf{x} \\
\therefore \frac{\partial f}{\partial \mathbf{x}}&=&
2A^T(A\mathbf{x-y})\\
\end{eqnarray}
$$
## 2
### (a)
$$
\begin{eqnarray}
f(\mathbf{x})&=&\sum_{k}(\sigma_{i}^2+x_{k}^2) \\
\therefore \frac{\partial f}{\partial x_{i}}&=&2x_{i}
\end{eqnarray}
$$
### (b)
$$
\begin{eqnarray}
\text{d}f&=&\text{d}tr(\sigma^2I+\mathbf{xx}^T) \\
&=&tr(\text{d}\mathbf{xx}^T+\mathbf{x}\text{d}\mathbf{x} ^T) \\
&=&tr(2\mathbf{x}^T\text{d}\mathbf{x}) \\
\therefore \frac{\partial f}{\partial \mathbf{x}}&=&2\mathbf{x}
\end{eqnarray}
$$
## 3
### (a)
$$
\begin{eqnarray}
f(\mathbf{x})&=& \frac{1}{4}\|A-\mathbf{xx}^T\| \\
&=& \frac{1}{4}\sum_{i}\sum_{j}(A_{ij}-x_{i}x_{j})^2 \\
&=& \frac{1}{4}\sum_{i}\sum_{j}x_{i}^2x_{j}^2-2A_{ij}x_{i}x_{j}+A_{ij}^2 \\
\therefore \frac{\text{d}f}{x_{k}}&=& \frac{1}{4}\left[ \sum_{j}(2x_{k}x_{j}^2-2A_{kj}x_{j})\sum_{i}(2x_{i}^2x_{k}-2A_{ik}x_{i}) \right] \\
\frac{\text{d}f}{x_{i}} &=& \sum_{K}x_{k}^2x_{i}-A_{kj}x_{k}
\end{eqnarray}
$$
### (b)
$$
\begin{eqnarray}
f(\mathbf{x})&=& \frac{1}{4}\|A-\mathbf{xx}^T\| \\
&=& \frac{1}{4}tr((A-\mathbf{xx}^T)^T(A-\mathbf{xx}^T)) \\
\text{d}f&=& \frac{1}{4} tr((d\mathbf{xx}^T+\mathbf{x}\text{d}\mathbf{x}^T)(A-\mathbf{xx}^T)+(A-xx)^T(d\mathbf{xx}^T)+\mathbf{x}\text{d}\mathbf{x}^T) \\
&=& \frac{1}{2}tr((\mathbf{xx}^T-A)^T(\text{d}\mathbf{xx}^T+\mathbf{x}\text{d}\mathbf{x}^T)) \\
&=& tr([\mathbf{xx}^T-A\mathbf{x}]^T\text{d}\mathbf{x}) \\
\because A&\succeq&{0} \\
\therefore \frac{\partial f}{\partial \mathbf{x}} &=&
(\mathbf{xx}^T-A)\mathbf{x}
\end{eqnarray}
$$
# 1.1.1.2
## 1
### (a)
$$
\begin{eqnarray}
f(\mathbf{x})&=&\sum_{a}\sum_{b}g_{ab}x_{a}x_{b} \\
\therefore\frac{\partial f}{\partial g_{ij}}&=&x_{i}x_{j}
\end{eqnarray}
$$
### (b)
$$
\begin{eqnarray}
f(\mathbf{x})&=&\mathbf{x}^TQ\mathbf{x}=tr(\mathbf{x}^TQ\mathbf{x}) \\
\therefore \text{d}f&=&\text{d}tr(\mathbf{x}^TQ\mathbf{x})=tr(\mathbf{xx}^T\text{d}Q) \\
&=& tr((\mathbf{xx}^T)^T\text{d}Q) \\
\therefore \frac{\partial f}{\partial Q}&=& \mathbf{xx}^T
\end{eqnarray}
$$
## 2
$$
\begin{eqnarray}
\text{d}f(\mathbf{x})&=&\text{d}tr(A\mathbf{X}B)=tr(A\text{d}\mathbf{x}B)=tr(BA\text{d}\mathbf{x}) \\
&=& tr((BA^T)^T\text{d}\mathbf{x})=tr((A^TB^T)^T\text{d}\mathbf{x}) \\
\therefore \frac{\partial f}{\partial \mathbf{x}}&=&A^TB^T
\end{eqnarray}
$$
# 1.1.1.3
## 1
$$
\begin{eqnarray}
f(\mathbf{x})&=&\sin \log(1+\mathbf{x}^T\mathbf{x})=tr(\sin \log(1+\mathbf{x}^T\mathbf{x})) \\
\text{d}f&=&\text{d}tr(\sin \log(1+\mathbf{x}^T\mathbf{x})) \\
&=& tr\left( \cos \log(1+\mathbf{x}^T\mathbf{x})\cdot \frac{\mathbf{x}^T\text{d}\mathbf{x}+\text{d}\mathbf{x}^T\mathbf{x}}{(1+\mathbf{x}^T\mathbf{x})\ln{2}} \right) \\
\because \mathbf{xx}^T&=& \|\mathbf{x}\|^2 \\
\therefore\text{d}f&=&\cos \log(1+\mathbf{x}^T\mathbf{x})\cdot \frac{1}{(1+\mathbf{x}^T\mathbf{x})\ln{2}}\cdot tr(\mathbf{x}^T\text{d}\mathbf{x}+\text{d}\mathbf{x}^T\mathbf{x}) \\
&=&2\cos \log(1+\mathbf{x}^T\mathbf{x})\cdot \frac{1}{(1+\mathbf{x}^T\mathbf{x})\ln{2}}\cdot tr(\mathbf{x}^T\text{d}\mathbf{x}) \\
\therefore \frac{\partial f}{\partial \mathbf{x}} &=&
\frac{2}{(1+\mathbf{x}^T\mathbf{x})\ln2}\cos \log(1+\mathbf x^T\mathbf{x})\cdot \mathbf{x}
\end{eqnarray} 
$$
# 1.1.2
## 1
$$
\begin{array}{ll}&\ \theta \mathbf{x}^T Q\theta \mathbf{x}+(1-\theta)\mathbf{y}^TQ(1-\theta)\mathbf{y}
\\ &=\theta ^2\mathbf{x}^T Q \mathbf{x}+(1-\theta)^2\mathbf{y}^T Q \mathbf{y} 
\\ & \le \theta ^2+(1-\theta)^2
\\ &= 2(\theta-\frac{1}{2} )^2+\frac{1}{2} 
\\ \\ &\because \theta \in \left [ 0,1 \right ] 
\\ \\ &\therefore \theta \mathbf{x}^T Q\theta \mathbf{x}+(1-\theta)\mathbf{y}^TQ(1-\theta)\mathbf{y}\le 1
\\ \\ &\therefore \theta \mathbf{x}+(1-\theta)\mathbf{y} \in S
\end{array}
$$
所以 $S$ 是凸集。
## 2
$$
\begin{array}{ll}& \nabla f(\mathbf{x})^T = \mathbf{x}^T(Q+Q^T)
\\ \\ &\therefore f(\mathbf{y})-f(\mathbf{x})- \nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})
\\ &=\mathbf{y}^TQ\mathbf{y}-\mathbf{x}^TQ\mathbf{x}-\mathbf{x}^T(Q+Q^T)(\mathbf{y}-\mathbf{x})
\\ &=(\mathbf{y}-\mathbf{x})^TQ(\mathbf{y}-\mathbf{x})
\\ & \ge 0
\end{array}
$$
所以 $f$ 为凸函数
## 3
构造 $g(t)=f(\mathbf{x}+t\mathbf{v})$，则仅需证明 $g(t)$ 为凸函数
$$
\begin{array}{ll}&\\ &\therefore g(t)\\ &=\left \| \mathbf{x}+t\mathbf{v} \right \|^2 \\ &=(\mathbf{x}+t\mathbf{v})^T(\mathbf{x}+t\mathbf{v})
\\ &=\mathbf{x}^T\mathbf{x}+t(\mathbf{x}^T\mathbf{v}+\mathbf{v}^T\mathbf{x})+t^2\mathbf{v}^T\mathbf{v}
\\ &=\mathbf{v}^T\mathbf{v}t^2+2\mathbf{x}^T\mathbf{v}t+\mathbf{x}^T\mathbf{x}
\\ \\ &\forall t_1,t_2,g(\frac{t_1+t_2}{2})-\frac{g(t_1)+g(t_2)}{2} 
\\ &=\mathbf{v}^T\mathbf{v}\left [ \frac{(t_1+t_2)^2}{4} -\frac{t_1^2+t_2^2}{2}  \right ] 
\\ &=\mathbf{v}^T\mathbf{v}\frac{2 t_1 t_2-(t_1^2+t_2^2)}{4} \\ &\le 0
\end{array}
$$
所以 $g(t)$ 为凸函数，即 $f(\mathbf{x})$ 为凸函数。
## 4
$$
\begin{array}{ll}& f(\mathbf{x})
\\ &=\frac{1}{2}||A\mathbf{x}-\mathbf{y}||\\ &=\frac{1}{2}(A\mathbf{x}-\mathbf{y})^T(A\mathbf{x}-\mathbf{y})
\\ \\ &\therefore \text{d}f
\\ &=\frac{1}{2} \text{tr}\left [ (A\text{d}\mathbf{x})^T(A\mathbf{x}-\mathbf{y})+(A\mathbf{x}-\mathbf{y})^TA\text{d}\mathbf{x} \right ] 
\\ &=\text{tr}\left [(A\mathbf{x}-\mathbf{y})^TA\text{d}\mathbf{x} \right ]
\\ \\ &\therefore \nabla f(\mathbf{x})=A^T(A\mathbf{x}-\mathbf{y})^T
\\ \\ &\therefore \text{d}\nabla f(\mathbf{x})=A^TA\text{d}\mathbf{x}=(A^TA)^T\text{d}\mathbf{x}
\\ \\ &\therefore \nabla^2 f(\mathbf{x})=A^TA
\\ \\ &\because A\text{为列满秩矩阵}
\\ \\ &\therefore A^TA
\\ &=\sum_{i}^{} \sum_{j}^{} A_{ij}^TA_{ji}\\ &=\sum_{i}^{}\sum_{j}^{} A_{ij}^2 \\ &\ge 0
\\ \\ &\therefore A\text{为正定矩阵},f(\mathbf{x})\text{为严格凸函数}
\end{array} 
$$
# 1.1.3
## 1
由题可得：
$$
\begin{eqnarray}
\nabla f(\mathbf{x})&=&(\mathbf{xx}^T-A)\mathbf{x}=0,
\\ 
A\mathbf{x}&=&\mathbf{xx}^T\mathbf{x}=\|\mathbf{x}\|^2\mathbf{x} \\
\therefore \mathbf{x}&=&0 \\
f(\mathbf{x})&=&\text{d}\mathbf{xx}^T\mathbf{x}+\mathbf{x\text{d}\mathbf{x}^T+\mathbf{xx}^T\text{d}\mathbf{x}-A\text{d}\mathbf{x}} \\
&=&\mathbf{x}^T\mathbf{x}\text{d}\mathbf{x}+\mathbf{xx}^T\text{d}\mathbf{x}+\mathbf{xx}^T\text{d}\mathbf{x}-A\text{d}\mathbf{x} \\
&=& (2\mathbf{xx}^T+\mathbf{x}^T\mathbf{x}-A)\text{d}\mathbf{x} \\
\therefore \nabla^2f(\mathbf{x})&=&2\mathbf{xx}^T+\mathbf{x}^T\mathbf{x}-A \\
if\,\mathbf{x}=0&,&\nabla f(\mathbf{x})= -A\prec 0 ,\,not\\
if\,\mathbf{xx}^T=A&,& \nabla^2f(\mathbf{x})=A+\mathbf{x}^T\mathbf{x} \\
\because &A&\text{为对称正定矩阵} \\
\therefore \nabla^2f(\mathbf{x})&=&\sum^{n}_{i=1}\lambda_{i}\mathbf{u}_{i}\mathbf{u}_{i}^T+\mathbf{x}_{i}^2 \succ 0 \\
if\,\mathbf{xx}^T=A&,& \mathbf{x}\text{为极小值点}
\end{eqnarray}
$$
## 2
$$
\begin{eqnarray}
\text{d}f(\mathbf{w}) &=& \frac{1}{N}\sum^{N}_{i=1}\ell{'}(\sigma(\mathbf{w}^T\mathbf{x}_{i})\mathbf{y}_{i})\sigma'(\mathbf{w}^T\mathbf{x}_{i})\text{d}\mathbf{w}^T\mathbf{x}_{i}+ \frac{1}{2}(\text{d}\mathbf{w}^T\mathbf{w}+\mathbf{w}^T\text{d}\mathbf{w}) \\
&=& \frac{1}{N}\sum^{N}_{i=1}\ell{'}(\sigma(\mathbf{w}^T\mathbf{x}_{i})\mathbf{y}_{i})\sigma'(\mathbf{w}^T\mathbf{x}_{i})\mathbf{x}_{i}\text{d}\mathbf{w}^T+\mathbf{w}^T\text{d}\mathbf{w} \\
\therefore \nabla f(\mathbf{w})&=&\frac{1}{N}\sum^{N}_{i=1}\ell{'}(\sigma(\mathbf{w}^T\mathbf{x}_{i})\mathbf{y}_{i})\sigma'(\mathbf{w}^T\mathbf{x}_{i})\mathbf{x}_{i}+\mathbf{w} \\
if\,\nabla f(\mathbf{w})=0&,&\mathbf{w}=\frac{1}{N}\sum^{N}_{i=1}\ell{'}(\sigma(\mathbf{w}^T\mathbf{x}_{i})\mathbf{y}_{i})\sigma'(\mathbf{w}^T\mathbf{x}_{i})\mathbf{x}_{i} \\
\mathbf{w}^T\mathbf{x}_{i} \, is \, a \, number&,&\therefore \ell{'}(\sigma(\mathbf{w}^T\mathbf{x}_{i})\mathbf{y}_{i})\sigma'(\mathbf{w}^T\mathbf{x}_{i})\,is\,a\,scalar.\\
let\,\alpha_{i}&=& -\frac{1}{N} \ell{'}(\sigma(\mathbf{w}^T\mathbf{x}_{i})\mathbf{y}_{i})\sigma'(\mathbf{w}^T\mathbf{x}_{i}) \\
\therefore \mathbf{w}&=&\sum_{i=1}^{N}\alpha_{i}\mathbf{x}_{i}
\end{eqnarray}
$$
# 1.1.4
## 1
### (a)
$$
\begin{align}
f(\mathbf x) &= \mathbf x^T \mathit Q \mathbf x \\
&= \mathbf x^T \mathit U\Lambda \mathit U^T \mathbf x\\
&= \mathbf x^T(\sum_{i=1}^{n}\lambda_i \mu_i \mu_i^t)\mathbf x\\
&= \sum_{i=1}^{n} \lambda_i (\mathbf x^T \mu_i)^2\\
\because&\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0 \mbox{ and } \|\mu_i\|^{2}=1 \\
\therefore&\mathbf x^T\mathit Q\mathbf x \le \lambda _1 \sum_{i=1}^{n}(\mathbf x^T \mu_i)^2\\
if\, ||&\mathbf x||=1, \mathbf x^T \mathit Q \mathbf x \le \lambda_1||\mathbf x||^2||\mu||^2=\lambda_1\\
\therefore &\lambda_1 = \underset{||\mathbf x|| = 1}{max} \quad \mathbf x^T \mathit Q \mathbf x = \underset{||\mathbf x||=1}{max} \quad \frac{\mathbf x^T \mathit Q \mathbf x}{\mathbf x^T \mathbf x}\\
if\, \mathbf x  &\ne 0, \mathbf x^T \mathit Q \mathbf x \le \lambda_1 ||\mathbf x||^2 ||\mu||^2\\
\because &||\mu||^2 = 1,||\mathbf x||^2 = \mathbf x^T \mathbf x\\
\therefore &\lambda_1 = \underset{x \ne 0}{max}\quad \frac{\mathbf x^T \mathit Q \mathbf x}{\mathbf x^T \mathbf x}\\
\therefore &\lambda_1 = \underset{||\mathbf x|| = 1}{max} \quad \mathbf x^T \mathit Q \mathbf x=\underset{x \ne 0}{max}\quad \frac{\mathbf x^T \mathit Q \mathbf x}{\mathbf x^T \mathbf x}
\\
\end{align}
$$
### (b)
$$
\begin{align}
f(\mathbf x&) = \mathbf x^T \mathit Q \mathbf x = \mathbf x^T \mathit U \Lambda  \mathit U^T \mathbf x\\
\mathbf y &= \mathit U^T \mathbf x\\
\because &\mathbf x^T \mathit Q \mathbf x = \mathbf y^T \Lambda \mathbf y = \sum_{i}^{n}\lambda_i \mathbf y_i^2\\
\because &\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0\\
\therefore &\sum_{i}^{n} \lambda_n \mathbf y_i^2 \le \mathbf x^T \mathit Q \mathbf x \le \sum_{i}^{n} \lambda_i \mathbf y^2\\
\because &\mathbf y_i^2 = (\mathit U^T\mathbf x)^2 且 ||\mu_i||=1\\
\therefore &\sum_{i}^{n}\mathbf y_i^2 \le ||\mathbf x||^2||\mathit U||^2 = ||\mathbf x||^2\\
\therefore &\lambda_n||\mathbf x||^2 \le \mathbf x^T \mathit Q \mathbf x \le \lambda_1 ||\mathbf x||^2
\end{align}
$$
# 1.1.5
## 1
必要性：因为 $f$ 是 $U-$ 强突的，所以可知 $\exists m>0$，使得 $h(\mathbf{x})=f(\mathbf{x})-\frac{m\left \| \mathbf{x} \right \|^2 }{2}$ 为凸函数。
充分性：
因为 $h(\mathbf{x})$ 为凸函数，所以 
$$
\begin{eqnarray}
h(\lambda \mathbf{x}+(1-\lambda)\mathbf{y})&=&f(\lambda \mathbf{x}+(1-\lambda) \mathbf{y})-\frac{\mu}{2} \left \| \lambda \mathbf{x}+(1-\lambda) \mathbf{y} \right \| ^2 \\
&\leq& \lambda h(\mathbf{x})+(1-\lambda)h(\mathbf{y}) \\
&=& \lambda\left [ f(\mathbf{x}) -\frac{\mu}{2} \left \| \mathbf{x} \right \| ^2\right ]+(1-\lambda )\left [ f(\mathbf{y}) -\frac{\mu}{2} \left \| \mathbf{y} \right \| ^2\right ]
\end{eqnarray}
$$
要证 $f$ 为 $U-$ 强凸，即证：
$$
\begin{eqnarray}
&f(\lambda \mathbf{x}+(1-\lambda \mathbf{y})) \le \lambda f(\mathbf{x})+(1-\lambda)f(\mathbf{y})-\frac{\mu }{2} \lambda (1-\lambda )\left \| \mathbf{x}-\mathbf{y} \right \|^2 
\\ &\therefore \lambda ||\mathbf{x}||^2+(1-\lambda )||\mathbf{y}||^2-||\mathbf{x}+(1-\lambda) \mathbf{y} ||^2
\\ & \ge \lambda (1-\lambda )\left \| \mathbf{x}-\mathbf{y} \right \|^2
\\ &=\lambda (1-\lambda ) \left [ ||\mathbf{x}||^2 +||\mathbf{y}||^2-2\mathbf{x}^T\mathbf{y}\right ] 
\\ &\text{即}\lambda ||\mathbf{x}||^2+(1-\lambda )||\mathbf{y}||^2 - \lambda ^2||\mathbf{x}||^2 - (1-\lambda )^2||\mathbf{y}||^2 - 2\lambda (1-\lambda )\mathbf{x}^T\mathbf{y}
\\ &\ge \lambda (1-\lambda ) \left [ ||\mathbf{x}||^2 +||\mathbf{y}||^2-2\mathbf{x}^T\mathbf{y}\right ] 
\end{eqnarray}
$$
该不等式恒成立，证毕
## 2
$f$ 是光滑的，所以：

$$
\begin{array}{ll}
& f(\mathbf{y})\le f(\mathbf{x})+\nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})+\frac{L}{2}||\mathbf{y}-\mathbf{x}||^2 
\\ & \nabla h(\mathbf{x})^T=L\mathbf{x}^T-\nabla f(\mathbf{x})^T
\\ &\therefore h(\mathbf{y})- h(\mathbf{x}) - \nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})
\\ &=\frac{L}{2}||\mathbf{y}||^2 -f(\mathbf{y})-\frac{L}{2}||\mathbf{x}||^2+f(\mathbf{x})-\left [ L\mathbf{x}^T-\nabla f(\mathbf{x})\right ](\mathbf{y}-\mathbf{x})
\\ &=\frac{L}{2}(||\mathbf{y}||^2+||\mathbf{x}||^2-2\mathbf{x}^T\mathbf{y})+L\mathbf{x}^T\mathbf{x} +\nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})+f(\mathbf{x})-f(\mathbf{y})
\\ &=\frac{L}{2}(||\mathbf{y}||^2+||\mathbf{x}||^2-2\mathbf{x}^T\mathbf{y})+\nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})+f(\mathbf{x})-f(\mathbf{y})
\\ &=\frac{L}{2}\left \| \mathbf{y}-\mathbf{x} \right \|^2 +\nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})+f(\mathbf{x})-f(\mathbf{y}) \\&\ge 0 \\
\end{array}
$$
所以 $h(\mathbf{x})$ 是凸函数，因过程可逆，故得证

## 3
### (a)
构造函数 $h(\mathbf{x})=\frac{L}{2}\left \| \mathbf{x} \right \|^2-f(\mathbf{x})$
$$
\begin{array}{ll}
&\therefore \nabla h(\mathbf{x})=L\mathbf{x}-\nabla f(\mathbf{x})
\\ &\therefore \nabla^2 h(\mathbf{x})=LI-\nabla^2 f(\mathbf{x})\ge 0 \\
\end{array} 
$$
所以 $h(\mathbf{x})$ 为凸函数且 $L=\max\lambda_{i}$ (最大特征值)
因此  $f(\mathbf{x})$ 是光滑的
构造函数 $h(\mathbf{x})=f(\mathbf{x}) -\frac{\mu}{2}\left \| \mathbf{x} \right \|^2$ 所以：
$$
\begin{array}{ll}
\\ &\therefore \nabla h(\mathbf{x})=\nabla f(\mathbf{x})-\mu\mathbf{x}
\\ &\therefore \nabla^2 h(\mathbf{x})=\nabla^2 f(\mathbf{x})-\mu I\ge 0
\end{array} 
$$
所以 $h(\mathbf{x})$ 为凸函数且 $\mu=\min\lambda_{i}$ (最大特征值)
因此  $f(\mathbf{x})$ 是强突的，且 $L=\max\lambda_{i},\mu=\min\lambda_{i}$
# 1.1.6
## 1
$$
\begin{eqnarray}
f(\mathbf{X}_{k+1})&=& \frac{1}{2}(\alpha\nabla f(\mathbf{X}_{k}))^TQ(\mathbf{X}_{k}-\alpha\nabla f(\mathbf{X}_{k})) \\
&=& \frac{1}{2}\mathbf{X}_{k}^TQ\mathbf{X}_{k+1}-\alpha\nabla f(\mathbf{X}_{k})Q\mathbf{X}_{k} \\
&=& f(\mathbf{X}_{k})-\alpha\nabla f(\mathbf{X}_{k})^TQ\mathbf{X}_{k}+ \frac{1}{2}\alpha^2\nabla f(\mathbf{X}_{k})^TQ\nabla f(\mathbf{X}_{k}) \\
\because f(\mathbf{X}_{k+1}) &<& f(\mathbf{X}_{k})
\therefore \alpha\nabla f(\mathbf{X}_{k})^TQ\mathbf{X}_{k}- \frac{1}{2}\alpha^2\nabla f(\mathbf{X}_{k})^TQ\nabla f(\mathbf{X}_{k})>0 \\
\because f(\mathbf{X}) &=& \frac{1}{2}\mathbf{X}^TQ\mathbf{X} \\
\therefore \text{d}f(x)&=&\frac{1}{2}(\text{d}\mathbf{X}^TQ\mathbf{X}+\mathbf{X}^TQ\text{d}\mathbf{X})=\mathbf{X}^TQ\text{d}\mathbf{X} \\
\therefore \nabla f(\mathbf{X})&=&Q\mathbf{X} \\ 
let\, g_{k}&=&\mathbf{X}Q\mathbf{X}_{K} \\
\alpha \mathbf{X}_{k}&Q^T&\mathbf{Q}\mathbf{X}_{k}-\frac{1}{2}\alpha^2\mathbf{X}_{k}^TQ^TQQ\mathbf{X}_{k}&>&0 \\
\because \alpha&>&0 \\
\therefore 0<&\alpha&<\frac{2g_{k}^Tg_{k}}{g_{k}^TQg_{k}}
\end{eqnarray}
$$
## 2
$$
\begin{eqnarray}
&\lambda_{1}=10,\lambda_{2}=1,k=\frac{\lambda_{1}}{\lambda_{2}}=10. & \\
&f(\mathbf{X})=\frac{1}{2}\mathbf{X}^TQ\mathbf{X}+10,
\nabla f(\mathbf{X})=Q\mathbf{X},\nabla f(\mathbf{X})=0\to x^{*}=0 & \\
\end{eqnarray}
$$
设 $\mathbf{d}_{k}$ 为下降方向，则新迭代点 $\mathbf{X}_{k+1}=X_{k}+ \alpha\mathbf{d}_{k}$
$$
\begin{eqnarray}
let:\min_{\alpha}h(\alpha)&=&f(\mathbf{X}_{k}+\alpha \mathbf{d}_{k}) \\
&=&\frac{1}{2}(\mathbf{X}_{k}+\alpha \mathbf{d}_{k})^TQ(\mathbf{X}_{k}+\alpha \mathbf{d}_{k})+10 \\
&=&\frac{1}{2}\mathbf{X}_{k}^TQ\mathbf{X}_{k}+\alpha \mathbf{d}_{k}^TQ\mathbf{x}_{k}+ \frac{1}{2}\alpha^2\mathbf{d}_{k}^TQ\mathbf{d}_{k} \\
&=& \frac{1}{2} \alpha^2\mathbf{d}_{k}^TQ\mathbf{d}_{k}+\alpha \mathbf{d}_{k}^TQ\mathbf{X}_{k}+f(\mathbf{x}_{k}) \\
&=& \frac{1}{2}\alpha^2\mathbf{d}_{k}^TQ\mathbf{d}_{k}+\alpha \mathbf{d}_{k}^T\nabla f(\mathbf{X}_{k})+f(\mathbf{X}_{k}) \\
h'(\alpha)&=&\mathbf{d}_{k}^T Q\mathbf{d}_{k}\alpha+\mathbf{d}_{k}^T\nabla f(\mathbf{X}_{k}) \\
\because h'(\alpha)&=&0 \\
\therefore \alpha&=& \frac{-\mathbf{d}_{k}^T\nabla f(\mathbf{X}_{k})}{\mathbf{d}_{k}^TQ\mathbf{d}_{k}} \\
f(\mathbf{X}_{k+1})&=&f(\mathbf{X}_{k})- \frac{\mathbf{d}_{k}^T\nabla f(\mathbf{X}_{k})}{\mathbf{d}_{k}^TQ\mathbf{d}_{k}}\mathbf{d}_{k} \\
\end{eqnarray}
$$
当 $\mathbf{d}_{k}=-\nabla f(\mathbf{X}_{k})$ 时：
$$
\alpha=\frac{\|\nabla f(\mathbf{X}_{k})\|^2}{\|\nabla f(\mathbf{X}_{k})\|_{Q}^2}
$$
所以
$$
\mathbf{X}_{k+1}=\mathbf{X}_{k}-\frac{\|\nabla f(\mathbf{X}_{k})\|^2}{\|\nabla f(\mathbf{X}_{k})\|_{Q}^2} \nabla f(\mathbf{X}_{k})
$$
由 $f(\mathbf{X}^*)=\frac{1}{2}\mathbf{X}^TQ\mathbf{X}^*+10=10$ ，由定理得，
$$
\begin{eqnarray}
f(\mathbf{X}^*_{k+1})-f(\mathbf{X}^*)&\leq&\left( 1- \frac{2}{1+k} \right)^2(f(\mathbf{X}_{k})-f(\mathbf{X}^*))    \\
&\leq&\left( 1- \frac{2}{1+k} \right)^{2t}(f(\mathbf{X}_{0})-f(\mathbf{X}^*))
\end{eqnarray}
$$
令 $R_{0}=f(\mathbf{x}_{0})-f(\mathbf{x}^*)$，则
$$
\begin{eqnarray}
R_{0}&=& \frac{1}{2}\mathbf{X}
(1,1)\begin{pmatrix}1 & 0\\0&10\end{pmatrix}\begin{pmatrix}
1 \\1\end{pmatrix}+10-10=5.5 \\
\therefore ( 1&-& \frac{2}{1+k} )^{2t}R_{0}<\epsilon=10^{-10} \\
\therefore t&>& \frac{\log\frac{\epsilon}{R_{0}}}{2\log(1-\frac{2}{1+k})} \\
\because \epsilon&=&10^{-10},R_{0}=5.5,k=10 \\
\therefore t&>&61.61
\end{eqnarray}
$$
最少62次
# 1.1.7
## 1
$$
\begin{eqnarray}
\tilde{\mathbf x_i} &=& arg\min_{\mathbf y \in L} \|\mathbf x_i - \mathbf y||\\ 
& =& \min_{t}||\mathbf x_i-\bar{x} - t\mathbf w\|
\\
\min_{t}||\mathbf x_i-\bar{x} - t\mathbf w|| &=& \|\mathbf w||^2t^2-2t\mathbf w^T(\mathbf x_i-\bar x)+\|\mathbf x_i - \bar x||^2   \\
&=&\|\mathbf w||^2(t-\frac{\mathbf w^T(\mathbf x_i-\bar x)}{\|\mathbf w||^2})+constant  \\
\end{eqnarray}
$$
所以：
$$
\begin{eqnarray}
t^* &=& \frac{\mathbf w^T(\mathbf x_i-\bar x)}{\|\mathbf w\|^2}    \\
\end{eqnarray}
$$
所以：
$$
\begin{eqnarray}
\tilde x &=& \bar x + \frac{\mathbf w^T(\mathbf x_i -\bar x)}{\|\mathbf w\|^2}\mathbf w \\
 &=& \bar x+ \frac{\mathbf w^T(\mathbf x_i-\bar x)}{\mathbf w^T \mathbf w}\mathbf w   \\
		
\end{eqnarray}
$$
## 2

## 3

# 1.3.1
## 精确线搜索
```python
import numpy as np
import matplotlib.pyplot as plt

Q = np.array([[1, 0], [0, 10]], dtype="float32")
func = lambda x: 0.5 * np.dot(x.T, np.dot(Q, x)).squeeze() + 10
gradient = lambda x: np.dot(Q, x)


x_0 = np.array([0, 0]).reshape([-1, 1])


def gradient_descent(start_point, func, gradient, epsilon=0.01):
    assert isinstance(start_point, np.ndarray)
    global Q, x_0
    x_k_1, iter_num, loss = start_point, 0, []
    xs = [x_k_1]

    while True:
        g_k = gradient(x_k_1).reshape([-1, 1])
        if np.sqrt(np.sum(g_k ** 2)) < epsilon:
            break
        alpha_k = np.dot(g_k.T, g_k).squeeze() / (np.dot(g_k.T, np.dot(Q, g_k))).squeeze()
        x_k_2 = x_k_1 - alpha_k * g_k
        iter_num += 1
        xs.append(x_k_2)
        loss.append(float(np.fabs(func(x_k_2) - func(x_0))))
        if np.fabs(func(x_k_2) - func(x_k_1)) < epsilon:
            break
        x_k_1 = x_k_2
    return xs, iter_num, loss


x0 = np.array([1,1], dtype="float32").reshape([-1, 1])
xs, iter_num, loss = gradient_descent(start_point=x0, func=func, gradient=gradient, epsilon=1e-10)
print(xs[-1])
print(iter_num)
plt.style.use("seaborn-v0_8")
plt.figure(figsize=[12, 6])
plt.plot(loss)
plt.xlabel("# iteration", fontsize=12)
plt.ylabel("Loss: $|f(x_k) - f(x^*)|$", fontsize=12)
plt.yscale("log")
plt.show()

```
## 固定步长精确线搜索
```python
import numpy as np
import matplotlib.pyplot as plt

Q = np.array([[1, 0], [0, 10]], dtype="float32")
func = lambda x: 0.5 * np.dot(x.T, np.dot(Q, x)).squeeze() + 10
gradient = lambda x: np.dot(Q, x)


x_0 = np.array([0, 0]).reshape([-1, 1])


def gradient_descent_constant_step(start_point, func, gradient, epsilon=0.01):
    assert isinstance(start_point, np.ndarray)
    global Q, x_0
    x_k_1, iter_num, loss = start_point, 0, []
    xs = [x_k_1]

    while True:
        g_k = gradient(x_k_1).reshape([-1, 1])
        if np.sqrt(np.sum(g_k ** 2)) < epsilon:
            break
        alpha = 1/10
        x_k_2 = x_k_1 - alpha * g_k
        iter_num += 1
        xs.append(x_k_2)
        loss.append(float(np.fabs(func(x_k_2) - func(x_0))))
        if np.fabs(func(x_k_2) - func(x_k_1)) < epsilon:
            break
        x_k_1 = x_k_2
    return xs, iter_num, loss


x0 = np.array([1,1], dtype="float32").reshape([-1, 1])
xs, iter_num, loss = gradient_descent_constant_step(start_point=x0, func=func, gradient=gradient, epsilon=1e-10)
print(xs[-1])
print(iter_num)
plt.style.use("seaborn-v0_8")
plt.figure(figsize=[12, 6])
plt.plot(loss)
plt.xlabel("# iteration", fontsize=12)
plt.ylabel("Loss: $|f(x_k) - f(x^*)|$", fontsize=12)
plt.yscale("log")
plt.show()

```
## 调用测试（Wolf）
```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

Q = np.array([[1, 0], [0, 10]], dtype="float32")
func = lambda x: 0.5 * np.dot(x.T, np.dot(Q, x)).squeeze() + 10
gradient = lambda x: np.dot(Q, x)


x_0 = np.array([0, 0]).reshape([-1, 1])


def gradient_descent_wolfe(start_point, func, gradient, epsilon=1e-6):
    assert isinstance(start_point, np.ndarray)
    global Q, x_0
    x_k_1, iter_num, loss = start_point, 0, []
    xs = [x_k_1]

    while True:
        g_k = gradient(x_k_1).reshape([-1, 1])
        if np.sqrt(np.sum(g_k ** 2)) < epsilon:
            break
        alpha_k = optimize.linesearch.line_search_wolfe2(f=func,myfprime=lambda x: np.reshape(np.dot(Q, x), [1, -1]),xk=x_k_1,pk=-g_k)[0]
        if alpha_k == None:
            break
        elif isinstance(alpha_k, float):
            alpha_k = alpha_k
        else:
            alpha_k = alpha_k.squeeze()

        x_k_2 = x_k_1 - alpha_k * g_k
        iter_num += 1
        xs.append(x_k_2)
        loss.append(float(np.fabs(func(x_k_2) - func(x_0))))
        if np.fabs(func(x_k_2) - func(x_k_1)) < epsilon:
            break
        x_k_1 = x_k_2
    return xs, iter_num, loss


x0 = np.array([1,1], dtype="float32").reshape([-1, 1])
xs, iter_num, loss = gradient_descent_wolfe(start_point=x0, func=func, gradient=gradient, epsilon=1e-10)
print(xs[-1])
print(iter_num)
plt.style.use("seaborn-v0_8")
plt.figure(figsize=[12, 6])
plt.plot(loss)
plt.xlabel("# iteration", fontsize=12)
plt.ylabel("Loss: $|f(x_k) - f(x^*)|$", fontsize=12)
plt.yscale("log")
plt.show()

```
# 1.3.2
```python
import numpy as np


n = 8 
B = np.random.rand(n, n)
A = np.dot(B, B.transpose())
x0 = np.random.rand(n,1)

print(x0)
print(A)
x = [x0] 
count = 0
size = 60
while count <= size:
    y = np.dot(A,x[-1])
    print(x[count])
    x.append(y/np.linalg.norm(y, ord=None, axis=None, keepdims=False))
    count = count + 1

lam = np.linalg.eig(A)
index = np.argmax(lam[0])
lamda_max = np.real(lam[0][index])
vector = lam[1][:,index]
vector_final = np.transpose((np.real(vector)))
print(lamda_max, vector_final)
print(vector_final-x[-1])

```