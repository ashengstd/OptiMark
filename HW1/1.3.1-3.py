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