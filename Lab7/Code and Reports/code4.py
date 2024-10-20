import matplotlib.pyplot as plt
import numpy as np

def leastSquareErrorPolynomial(matrix,m,plot_poly = False):
    if m >= len(matrix):
        m = len(matrix)-1
    x = []
    y = []
    n = len(matrix)
    for i in range(n):
        x.append(matrix[i][0])
        y.append(matrix[i][1])
    y = np.array(y)
    A = np.zeros((n,m+1))

    for i in range(n):
        for j in range(m+1):
            A[i][j] = x[i]**j
    A_T = np.transpose(A)
    ATA = np.dot(A_T,A)
    intermediate = np.dot(np.linalg.inv(ATA),A_T)
    coef = np.dot(intermediate,y)
    coef = coef[::-1]
    polynomial = np.poly1d(coef)
    if plot_poly:
        plot_fun(x,y,polynomial)
    return coef,polynomial
def plot_fun(x,y,polynom):
        err = 1e-3
        plt.scatter(x,y,label = 'Points')
        n = len(x)
        x_range = np.arange(min(x)-0.1,max(x)+0.1,err)
        y_pred = polynom(x_range)
        plt.plot(x_range,y_pred,label = 'Least Square Error Polynomial')
        # if self.function is not None:
        #     plt.plot(x_range,self.function(x_range),label = 'Original Function')
        plt.legend()
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot()

x = [1,2,3,4,5]
y = [1,3,8,12,25]
mat = np.column_stack((x,y))
coef,poly_i = leastSquareErrorPolynomial(mat,m=3,plot_poly=True)
print('Coefficient of Least Square Error Polynomial : ',coef)

poly_lst = []
poly_coef = []
x_r = np.linspace(min(x)-0.1,max(x)+0.1,1000)
for i in range(1,len(x)):
    coef,poly_i = leastSquareErrorPolynomial(mat,i)
    poly_coef.append(coef)
    poly_lst.append(poly_i)
    plt.plot(x_r,poly_i(x_r),label = str(i) + ' Order')
plt.scatter(x,y,label = 'Points')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Square Error Polynomial with different Orders')
plt.plot()
