import numpy as np
import matplotlib.pyplot as plt
import copy
class LinearSolve:
    def __init__(self,matrix,vector) -> None:
        self.matrix = matrix
        self.vector = vector
        self.res = np.array([])
        self.err = 1e-7
    def get_upper_permute(self,matrix):
        n = len(matrix)
        upper = copy.deepcopy(matrix)
        permute = np.eye(n)
        for i in range(n):
            if(upper[i][i] == 0):
                j = i+1
                while(j < n and upper[j][i] == 0):
                    j += 1
                if(j == n):
                    continue
                else:
                    permute[[i,j]] = permute[[j,i]]
                    upper = np.dot(permute,upper)
            for j in range(i+1,n):
                m = upper[j][i]/upper[i][i]
                upper[j][i] = 0
                for k in range(i+1,n+1):
                    upper[j][k] = upper[j][k] - m*upper[i][k]
        return upper,permute
    
    def get_inv(self,matrix):
        matinv = np.linalg.inv(matrix)
        return matinv

    def get_lower(self,matrix,upper,permute):
        pa = np.dot(permute,matrix)
        upp_inv = self.get_inv(upper)
        return np.dot(pa,upp_inv)

    def gauss(self,A_n,b):
        n = len(A_n)
        A = copy.deepcopy(A_n)
        #st = deque()
        for i in range(n):
            A[i].append(b[i])
        
        A,_ = self.get_upper_permute(A)
        x = [0]*n
        for i in range(n-1,-1,-1):
            x[i] = A[i][n]
            k = 0
            for j in range(n-1,i,-1):
                x_j = x[j]*A[i][j]
                if(abs(x_j) < self.err):
                    continue
                x[i] -= x[j]*A[i][j]
            x[i] /= A[i][i]
            if(abs(x[i]) < self.err):
                x[i] = 0
        self.res = x
        return np.array(x)
    
    def vector_norm(self,x1,x2):
        return np.linalg.norm(np.array(x1) - np.array(x2))
    
    def jacobi(self,A,b,guess):
        C,permute = self.diagonal_dominance(A)
        if C == False:
            return None,None,None
        n = len(A)
        iter = []
        ite = 1
        norm = []
        x = copy.deepcopy(guess)
        mat = [0]*n
        norm_val = 0
        new_x = copy.deepcopy(x)
        while True:
            for i in range(n):
                sum_val = 0
                for j in range(n):
                    if i != j :
                        sum_val -= C[i][j]*x[j]
                new_x[i] = (b[i] + sum_val)/C[i][i]
            
            norm_val = self.vector_norm(new_x,x)
            # print(new_x)
            # print(x)
            if(norm_val < self.err):
                break
            iter.append(ite)
            ite += 1
            norm.append(norm_val)
            x = copy.deepcopy(new_x)
        
        new_x = copy.deepcopy(x)
        for i in range(n):
            new_x[i] = x[permute[i]]
        x = new_x
        return np.array(x),iter,norm
    
    def diagonal_dominance(self,A):
        flag = True
        n = len(A)
        column_idx = [0]*n
        uset = set()
        for i in range(n):
            column_idx[i] = A[i].index(max(A[i],key=abs))
            #print(column_idx[i])
            if column_idx[i] in uset:
                print('Matrix is not Diagonally Dominant and hence result cannot be computed')
                return False,column_idx
            val = 0
            for j in range(n):
                if j != column_idx[i]:
                    val += abs(A[i][j])
            if val >= abs(A[i][column_idx[i]]):
                print('Matrix is not Diagonally Dominant and hence result cannot be computed')
                return False,column_idx
            uset.add(column_idx[i])
        C = copy.deepcopy(A)
        for i in range(n):
            if i != column_idx[i]:
                flag = False
            C[column_idx[i]] = A[i]
        if not flag:
            print('Matrix is not Diagonally Dominant but permutation of rows can make it dominant')
        else:
            print('Matrix is Diagonally Dominant')
        return C,column_idx

    def seidal(self,A,b,guess):
        C,permute = self.diagonal_dominance(A)
        if C == False:
            return None,None,None
        n = len(A)
        x = copy.deepcopy(guess)
        iter = []
        norm = []
        ite = 1
        mat = [0]*n
        norm_val = 0
        old_x = copy.deepcopy(x)
        while True:
            for i in range(n):
                sum_val = 0
                for j in range(n):
                    if i != j :
                        sum_val -= C[i][j]*x[j]
                x[i] = (b[i] + sum_val)/C[i][i]
            #print(x)
            norm_val = self.vector_norm(x,old_x)
            if(norm_val < self.err):
                break
            iter.append(ite)
            ite += 1
            norm.append(norm_val)
            old_x = copy.deepcopy(x)
        new_x = copy.deepcopy(x)
        for i in range(n):
            new_x[i] = x[permute[i]]
        x = new_x
        return np.array(x),iter,norm
    
    def inbuilt(self,A_n,b):
        return np.linalg.solve(A_n,b)
    
    def get_roots(self,method = 'None',guess = np.array([])):
        match method:
            case 'gauss':
                return self.gauss(self.matrix,self.vector)
            case 'numpy':
                return self.inbuilt(self.matrix,self.vector)
            case 'seidal':
                return self.seidal(self.matrix,self.vector,guess)
            case 'jacobi':
                return self.jacobi(self.matrix,self.vector,guess)
            case 'None':
                return self.inbuilt(self.matrix,self.vector)

A = [[9,1,1],[2,10,3],[3,4,11]]
b = [10,19,0]
guess = [0,0,0]
ls = LinearSolve(A,b)
val_j,j_iter,j_norm = ls.get_roots(method = 'jacobi',guess = guess)
print('Roots are (jacobi method): ',val_j)
val_s,s_iter,s_norm = ls.get_roots(method = 'seidal',guess = guess)
print('Roots are (seidal method): ',val_s)
val_x = ls.get_roots(method='numpy')
print('Roots using numpy: ',val_x)
if val_j is not None:
    plt.plot(j_iter,j_norm,label = 'Jacobi Method')
    plt.plot(s_iter,s_norm,label = 'Seidal Method')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Distance between succesive values of x')
    plt.title('Convergence of Different Methods')
    plt.show()

A = [[2,10,3],[9,1,1],[3,4,11]]
b = [10,19,0]
guess = [0,0,0]
ls = LinearSolve(A,b)
val_j,j_iter,j_norm = ls.get_roots(method = 'jacobi',guess = guess)
print('Roots are (jacobi method): ',val_j)
val_s,s_iter,s_norm = ls.get_roots(method = 'seidal',guess = guess)
print('Roots are (seidal method): ',val_s)
val_x = ls.get_roots(method='numpy')
print('Roots using numpy: ',val_x)
if val_j is not None:
    plt.plot(j_iter,j_norm,label = 'Jacobi Method')
    plt.plot(s_iter,s_norm,label = 'Seidal Method')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Distance between succesive values of x')
    plt.title('Convergence of Different Methods')
    plt.show()

A = [[2,10,3],[9,9,1],[3,4,11]]
b = [10,19,0]
guess = [0,0,0]
ls = LinearSolve(A,b)
val_j,j_iter,j_norm = ls.get_roots(method = 'jacobi',guess = guess)
print('Roots are (jacobi method): ',val_j)
val_s,s_iter,s_norm = ls.get_roots(method = 'seidal',guess = guess)
print('Roots are (seidal method): ',val_s)
val_x = ls.get_roots(method='numpy')
print('Roots using numpy: ',val_x)
if val_j is not None:
    plt.plot(j_iter,j_norm,label = 'Jacobi Method')
    plt.plot(s_iter,s_norm,label = 'Seidal Method')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Distance between succesive values of x')
    plt.title('Convergence of Different Methods')
    plt.show()
