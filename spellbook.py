from scipy import linalg, linspace, array, dot, eye, empty, sqrt, exp, sin, cos
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from tqdm import tqdm
import abc
from colorama import Fore


# 0th Circle
def euler_explicit(funcs, x_start, t_stop, h=0.1):
    """
    Явный метод Эйлера

    :param funcs: функция, принимающая на вход вектор и возвращающая вектор. Функция должна принимать в качестве первого аргумента время.
    :param x_start: начальный вектор системы. Первая координата - начальное время
    :param t_stop: конечное время интегрирования
    :param h: шаг h
    :return: вектор состояния системы в конечный момент времени. Первая координата - конечное время
    """
    x = x_start.copy()
    y = [x_start.copy()]
    while x[0] < t_stop:
        x += h*funcs(x)
        y.append(x.copy())
    return np.array(y).T


# 1th Circle
def euler_implicit(funcs, x_start, t_stop, h=0.1):
    """
    Неявный метод Эйлера

    :param funcs: функция, принимающая на вход вектор и возвращающая вектор. Функция должна принимать в качестве первого аргумента время.
    :param x_start: начальный вектор системы. Первая координата - начальное время
    :param t_stop: конечное время интегрирования
    :param h: шаг h
    :return: вектор состояния системы в конечный момент времени. Первая координата - конечное время
    """
    x = x_start.copy()
    y = [x_start.copy()]
    while x[0] < t_stop:
        x_tmp_tmp = x + h*funcs(x)
        x_tmp = x + h*(funcs(x) + funcs(x_tmp_tmp))/2
        x += h*(funcs(x) + funcs(x_tmp))/2
        y.append(x.copy())
    return np.array(y).T


# 2th Circle
def euler_central_point(funcs, x_start, t_stop, h=0.1):
    """
    Метод Эйлера с центрольной точкой

    :param funcs: функция, принимающая на вход вектор и возвращающая вектор. Функция должна принимать в качестве первого аргумента время.
    :param x_start: начальный вектор системы. Первая координата - начальное время
    :param t_stop: конечное время интегрирования
    :param h: шаг h
    :return: вектор состояния системы в конечный момент времени. Первая координата - конечное время
    """
    x = x_start.copy()
    y = [x_start.copy()]
    while x[0] < t_stop:
        x_tmp = (funcs(x + h*funcs(x)) + funcs(x))/2
        x += h*x_tmp
        y.append(x.copy())
    return np.array(y).T


# 3rd Circle
def adams_2(funcs, x_start, t_stop, h=0.001):
    """
    Метод Адамса второго порядка

    :param funcs: функция, принимающая на вход вектор и возвращающая вектор. Функция должна принимать в качестве первого аргумента время.
    :param x_start: начальный вектор системы. Первая координата - начальное время
    :param t_stop: конечное время интегрирования
    :param h: шаг h
    :return: вектор состояния системы в конечный момент времени. Первая координата - конечное время
    """
    init_method = RungeExplicit(np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0, 0, 1]]))
    y = list(init_method(funcs, x_start.copy(), x_start[0]+h, h).T)
    x = y[-1]
    y = y[:-1]
    while x[0] < t_stop:
        x += h*((3/2)*funcs(x)-(1/2)*funcs(y[-1]))
        y.append(x.copy())
    return np.array(y).T


def adams_3(funcs, x_start, t_stop, h=0.001):
    """
    Метод Адамса третьего порядка

    :param funcs: функция, принимающая на вход вектор и возвращающая вектор. Функция должна принимать в качестве первого аргумента время.
    :param x_start: начальный вектор системы. Первая координата - начальное время
    :param t_stop: конечное время интегрирования
    :param h: шаг h
    :return: вектор состояния системы в конечный момент времени. Первая координата - конечное время
    """
    init_method = RungeExplicit(np.array([[0, 0, 0, 0], [0.5, 0.5, 0, 0], [1, 0, 1, 0], [0, 1/6, 2/3, 1/6]]))
    y = list(init_method(funcs, x_start.copy(), x_start[0]+2*h, h).T)
    x = y[-1]
    y = y[:-1]
    while x[0] < t_stop:
        x += h*((23/12)*funcs(x)-(16/12)*funcs(y[-1])+(5/12)*funcs(y[-2]))
        y.append(x.copy())
    return np.array(y).T


# 4rd Circle
def adams_4(funcs, x_start, t_stop, h=0.001):
    """
    Метод Адамса 4 порядка
    
    :param funcs: функция, принимающая на вход вектор и возвращающая вектор. Функция должна принимать в качестве первого аргумента время.
    :param x_start: начальный вектор системы. Первая координата - начальное время
    :param t_stop: конечное время интегрирования
    :param h: шаг h
    :return: вектор состояния системы в конечный момент времени. Первая координата - конечное время
    """
    init_method = RungeExplicit(np.array([[0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0], [0.5, 0, 0.5, 0, 0],
                                          [1, 0, 0, 1, 0], [0, 1/8, 3/8, 3/8, 1/8]]))
    y = list(init_method(funcs, x_start.copy(), x_start[0]+3*h, h).T)
    x = y[-1]
    y = y[:-1]
    while x[0] < t_stop:
        x += h*((55/24)*funcs(x)-(59/24)*funcs(y[-1])+(37/24)*funcs(y[-2])-(9/24)*funcs(y[-3]))
        y.append(x.copy())
    return np.array(y).T


# 5th Circle
# TODO
# To be inherited from Onestepmethod class

class RungeExplicit:
    
    """
    Класс с явными методами Рунге-Кутты
    """
    
    def __init__(self, butcher):
        """
        Создать явный метод РК с заданной таблицей Бутчера 
        :param butcher: таблица Бутчера
        """
        self.alphas = butcher[:-1, 0]
        self.order = len(self.alphas)
        self.gammas = butcher[-1, 1:]
        self.betas = butcher[:-1, 1:]

    def __call__(self, funcs, x_start, t_stop, h=0.1):
        """
        Проинтегрировать систему заданным методом РК

        :param funcs: функция, принимающая на вход вектор и возвращающая вектор. Функция должна принимать в качестве первого аргумента время.
        :param x_start: начальный вектор системы. Первая координата - начальное время
        :param t_stop: конечное время интегрирования
        :param h: шаг h
        :return: вектор состояния системы в конечный момент времени. Первая координата - конечное время
        """
        x = x_start.copy()
        y = [x_start.copy()]
        while x[0] < t_stop:
            K = []
            for i in range(self.order):
                addition = np.zeros(len(x))
                addition[0] = self.alphas[i]*h
                for j in range(len(K)):
                    addition[1:] += h*K[j]*self.betas[i, j]
                K.append(funcs(x+addition)[1:])
            for i in range(len(K)):
                x += np.array([0] + list(h*K[i]*self.gammas[i]))
            x += np.array([h] + [0]*(len(x)-1))
            y.append(x.copy())
        return np.array(y).T


# 6th Circle
def jacobian(f, t_0, y_0):
    jac = np.zeros((len(y_0), len(y_0)))
    for i in range(len(y_0)):
        def g(x):
            y = y_0.copy()
            y[i] = x
            return f(t_0, y)
        for j in range(len(y_0)):
            def h(x):
                return g(x)[j]
            jac[j][i] = derivative(h, y_0[i])

    return jac


class Onestepmethod(object):

    """
    Общий класс для одношаговых методов
    """

    def __init__(self, f: callable, y0: array, t0: float, te: float, n: int):

        """
        Создать одношаговый метод

        :param f: система уравнений
        :param y0: начальный вектор системы
        :param t0: начальное время интегрирования
        :param te: конечное время интегрирования
        :param N: число шагов
        """

        self.f = f
        self.y0 = y0.astype(float)
        self.t0 = t0
        self.interval = [t0, te]
        self.grid = linspace(t0, te, n+2)
        self.h = (te-t0)/(n+1)
        self.N = n
        self.m = len(y0)
        self.solution = None

    def step(self):

        """
        Провести одну итерацию интегрирования

        :return: вектор следующего состояния

        """
        ti, yi = self.grid[0], self.y0
        tim1 = ti
        yield np.hstack((array([ti]), yi))
        for ti in tqdm(self.grid[1:], bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            yi = yi + self.h * self.phi(tim1, yi)
            tim1 = ti
            yield np.hstack((array([ti]), yi))

    def solve(self):

        """
        Провести все шаги интегрирования до конечного момента времени
        :return: Список из промежуточных векторов состояния системы
        """

        self.solution = list(self.step())

    @abc.abstractmethod
    def phi(self, tim1, yi) -> array:
        """
        # TODO: а что это?
        :param tim1: ??
        :param yi: ??
        :return: ??
        """
        return array([])


# 7th Circle
class RungeImplicit(Onestepmethod):
    """
    Класс для неявных методов РК
    """

    def __init__(self, A, b, c, *args, tol=1e-9):
        super(RungeImplicit, self).__init__(*args)
        self.A = A
        self.b = b
        self.c = c
        self.s = len(self.b)
        self.tol = tol

    def phi(self, t0, y0):
        """
        # TODO: что это?

        :param t0: начальное время
        :param y0: начальный вектор состояния
        :return:
        """
        M = 10
        stageDer = array(self.s*[self.f(t0, y0)])
        J = jacobian(self.f, t0, y0)
        stageVal = self.phi_solve(t0, y0, stageDer, J, M)
        return array([dot(self.b, stageVal.reshape(self.s, self.m)[:, j]) for j in range(self.m)])

    def phi_solve(self, t0, y0, initVal, J, M):
        JJ = eye(self.s*self.m)-self.h*np.kron(self.A, J)
        luFactor = linalg.lu_factor(JJ)
        for i in range(M):
            initVal, norm_d = self.phi_newtonstep(t0, y0, initVal, luFactor)
            if norm_d < self.tol:
                # print('Newton converged in {} steps'.format(i))
                break
            elif i == M-1:
                raise ValueError('The Newton iteration did not converge.')
        return initVal

    def phi_newtonstep(self, t0, y0, initVal, luFactor):
        """
        Еще одна магия

        :param t0:
        :param y0:
        :param initVal:
        :param luFactor:
        :return:
        """
        d = linalg.lu_solve(luFactor, - self.F(initVal.flatten(), t0, y0))
        return initVal.flatten() + d, norm(d)

    def F(self, stageDer, t0, y0):
        """
        ??

        :param stageDer:
        :param t0:
        :param y0:
        :return:
        """
        stageDer_new = empty((self.s, self.m))
        for i in range(self.s):
            stageVal = y0 + array([self.h * dot(self.A[i, :],
                                  stageDer.reshape(self.s, self.m)[:, j]) for j in range(self.m)])
            stageDer_new[i, :] = self.f(t0 + self.c[i] * self.h, stageVal)
        return stageDer - stageDer_new.reshape(-1)


class Gauss(RungeImplicit):

    """
    Класс с методом Гаусса (частный случай неявного РК)
    """

    def __init__(self, *args):
        A = array([[5/36, 2/9 - sqrt(15)/15, 5/36 - sqrt(15)/30],
                 [5/36 + sqrt(15)/24, 2/9, 5/36 - sqrt(15)/24],
                 [5/36 + sqrt(15)/30, 2/9 + sqrt(15)/15, 5/36]])
        b = [5/18, 4/9, 5/18]
        c = [1/2-sqrt(15)/10, 1/2, 1/2+sqrt(15)/10]
        super(Gauss, self).__init__(A, b, c, *args)


class CROS(Onestepmethod):

    """
    Класс с методом Розенброка с комплексными коэффициентами
    """

    def phi(self, tim1, yi):
        jac = jacobian(self.f, tim1, yi)
        a = eye(self.m, self.m) - 0.5*(1+1j)*self.h*jac
        b = self.f(tim1+0.5*self.h, yi)
        lu = linalg.lu_factor(a)
        w = linalg.lu_solve(lu, b)
        return w.real


# TODO
# 8th Circle
# 9th Circle


if __name__ == '__main__':
    """
    Здесь идет тестирование методов на наличие серьезных косяков. Но, бог свидетель, как же это тяжело читать!
    """
    t0, te = 0, 10
    tol_sol = 1e-5

    def f(t, y):
        return array([-y[1],
                      y[0]-y[1]])

    init = array([0, sqrt(3)/2])

    # solution
    # array([t, -exp(-t/2)*sin(t*sqrt(3)/2), exp(-t/2)*((-1/2)*sin(t*sqrt(3)/2)+(sqrt(3)/2)*cos(t*sqrt(3)/2))])

    # def f(t, y):
    #     return array([[(1-2*t)*y[0]]])

    # init = array([1.])
    # solution
    # array([t, exp(t-t**2)])

    N = [2 * n for n in range(100)]
    stepsize = []
    mean_error = []
    result = []
    expected = []
    # TODO: возможно, стоит сделать тесты автоматизироваными?
    for n in N:
        stepsize.append((te - t0) / (n + 1))
        timeGrid = linspace(t0, te, n + 2)
        expected = [array([t, -exp(-t/2)*sin(t*sqrt(3)/2),
                           exp(-t/2)*((-1/2)*sin(t*sqrt(3)/2)+(sqrt(3)/2)*cos(t*sqrt(3)/2))]) for t in timeGrid]
        method = CROS(f, init, t0, te, n)
        method.solve()
        result = method.solution
        print(result)
        error = [norm(expected[i][1:] - result[i][1:]) for i in range(len(timeGrid))]
        mean = np.mean(error)
        mean_error.append(mean)
        print(mean, error[1])
    result = array(result)
    expected = array(expected)
    X, Y = result[:, 1].flatten(), result[:, 2].flatten()
    X_t, Y_t = expected[:, 1].flatten(), expected[:, 2].flatten()
    plt.plot(X, Y, 'bo', alpha=0.7)
    plt.plot(X_t, Y_t, 'r-')
    plt.legend(['predict', 'Ъ'])
    plt.show()
