import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
err = 1e-7

# %% [markdown]
# # Q1)
#  **$$f(x)=x^{6}-x-1$$**

# %%
def fun(x):
    return x**6 - x - 1

def dfun(x):
    return 6*(x**5) - 1

def newton_raphson(x0,err):
    data = []
    roots = []
    error = []
    present = x0
    fpresent = fun(x0)
    dfpresent = dfun(x0)
    next = present-(fpresent/dfpresent)
    roots.append(present)
    i = 1
    while(abs(next - present) > err):
        prev = present
        present = next
        fpresent = fun(present)
        dfpresent = dfun(present)
        next = present-(fpresent/dfpresent)
        data.append([i,present,fpresent,next,present - prev])
        roots.append(present)
        error.append(abs(present - prev))
        i += 1
    alpha = next
    data.append([i,next,fun(next),next - (fun(next)/dfun(next)),next - present])
    error.append(next- present)
    for i in range(0,len(data)):
        data[i].append(alpha - roots[i])
    
    df = pd.DataFrame(data,columns =['iter', 'x\u2099', 'f(x\u2099)', 'x\u2099\u208a\u2081', 'x\u2099 - x\u2099\u208b\u2081', 'a - x\u2099\u208b\u2081'])
    iter = np.arange(1,len(error)+1,1)
    plt.figure(1)
    plt.plot(iter,error,label = 'root = ' + str(roots[-1]))
    plt.legend()
    plt.xlabel('Iteration No.')
    plt.ylabel('Error')
    plt.title('Error vs Iteration Graph')
    plt.grid(True)
    plt.plot()
    plt.figure(2)
    r = [roots[-1]]*len(iter)
    plt.plot(iter,roots,label = '$x_n$')
    plt.plot(iter,r,label = 'root')
    plt.legend()
    plt.xlabel('Iteration No.')
    plt.ylabel('$x_n$')
    plt.title('Convergence of x towards the root')
    plt.grid(True)
    plt.plot()

    return df


# %%
df = newton_raphson(1.5,err)
df



# %%
df = newton_raphson(-1,err)
df

# %% [markdown]
# Result:
# 1. For the initial point 1.5 the root is  1.134724145316218.
# 2. For the initial point -1 the root is -0.7780895986786547.

# %% [markdown]
# # Q2)

# %% [markdown]
# **$$f(x) = x^{3}-x^{2}-x-1$$**

# %%
def fun(x):
    return x**3 - x**2 - x - 1
def dfun(x):
    return 3*x**2 - 2*x - 1
df = newton_raphson(2,err)
df

# %% [markdown]
# Result:
# 1. For the initial point 2,the root is 1.8392868100680193

# %% [markdown]
# **$$f(x) = 1 + 0.3*cos(x) $$**
# 

# %%
def fun(x):
    return 1 + 0.3*np.cos(x) - x
def dfun(x):
    return -1 - 0.3*np.sin(x)

df = newton_raphson(0,err)
df

# %% [markdown]
# Result:
# 1. For the initial point 0,the root is 1.1284251543001005.

# %% [markdown]
# **$$f(x)= sin(x) + 1/2 - cos(x)$$**

# %%
def fun(x):
    return 0.5 + np.sin(x) - np.cos(x)

def dfun(x):
    return np.cos(x) + np.sin(x)
df = newton_raphson(0,err)
df

# %% [markdown]
# Result:
# 1. For the initial point 0,the root is 0.42403103949074533

# %% [markdown]
# **$$ f(x) = e^{-x} - x$$**

# %%
def fun(x):
    return np.exp(-x) - x

def dfun(x):
    return -1 - np.exp(-x)
df = newton_raphson(1,err)
df

# %% [markdown]
# Result:
# 1. For the initial point 1,the root is 0.537143285989123

# %% [markdown]
# **$$ f(x) = e^{-x} - sin(x)$$**

# %%
def fun(x):
    return np.exp(-x) - np.sin(x)
def dfun(x):
    return -np.exp(-x) - np.cos(x)
df = newton_raphson(1,err)
df

# %% [markdown]
# Result:
# 1. For the initial point 1,the root is 0.5885327439585476

# %% [markdown]
# **$$ f(x) = x^3 - 2x - 2$$**

# %%
def fun(x):
    return x**3 - 2*x - 2
def dfun(x):
    return 3*x**3 - 2
df = newton_raphson(1,err)
df

# %% [markdown]
# Result:
# 1. For the initial point 1,the root is 1.7692925029979065

# %% [markdown]
# **$$ f(x) = x^4 - x - 1$$**

# %%
def fun(x):
    return x**4 - x - 1
def dfun(x):
    return 4*x**3 - 1
df = newton_raphson(1,err)
df

# %% [markdown]
# Result:
# 1. For the initial point 1,the root is 1.220744084605788

# %% [markdown]
# **$$ f(x) = tan(x) - x$$**

# %%
def fun(x):
    return np.tan(x) - x
def dfun(x):
    return (1/(np.cos(x)**2))- 1
df = newton_raphson(1,err)
df

# %% [markdown]
# Result:
# 1. For the initial point 1,the root is 2.7659680186998385e-07

# %% [markdown]
# # Q4)

# %% [markdown]
# **$$f(x) = a + x(x-1)^2 $$**

# %%
def newton_raphson(a,x0,err):
    roots = []
    present = x0
    fpresent = fun(a,x0)
    dfpresent = dfun(x0)
    next = present-(fpresent/dfpresent)
    roots.append(present)
    i = 1
    while(abs(next - present) > err):
        prev = present
        present = next
        fpresent = fun(a,present)
        dfpresent = dfun(present)
        next = present-(fpresent/dfpresent)
        roots.append(present)
        i += 1
    return roots

def fun(a,x):
    return a + x*((x-1)**2)
def dfun(x):
    return (x-1)**2 + 2*x*(x-1)

root0 = newton_raphson(0,-0.5,err)
root1 = newton_raphson(0,1.5,err)
r0 = [0]*len(root0)
r1 = [1]*len(root1)
iter0 = np.arange(1,len(root0)+1,1)
iter1 = np.arange(1,len(root1)+1,1)
plt.figure(1)
plt.plot(iter0,r0,label = 'Root = 0')
plt.plot(iter0,root0,label = 'Newton Raphson with Initial Guess = -0.5')
plt.legend()
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Value of root')
plt.title('Convergence towards root')
plt.plot()

plt.figure(2)
plt.plot(iter1,r1,label = 'Root = 1')
plt.plot(iter1,root1,label = 'Newton Raphson with Initial Guess = 1.5')
plt.legend()
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Value of root')
plt.title('Convergence towards root')
plt.plot()
a_val = [0,0.01,0.02,0.03,0.04,0.05,0.07,0.08,0.09,0.1]
roots0 = []
iter0 = []
roots1 = []
iter1 = []

for a in a_val:
    x = newton_raphson(a,0.99,err)
    roots0.append(x[-1])
    iter0.append(len(x))

for a in a_val:
    x = newton_raphson(a,1.01,err)
    roots1.append(x[-1])
    iter1.append(len(x))

plt.figure(3)
plt.plot(a_val,roots0,label = 'Initial Guess = ' + str(0.9),linewidth = 3)
plt.plot(a_val,roots1,label = 'Initial Guess = ' + str(1.1))
plt.legend()
plt.grid(True)
plt.xlabel('Value of a')
plt.ylabel('Negative Root Value')
plt.title('Convergence towards negative real root with increasing a')
plt.plot()

plt.figure(4)
plt.plot(a_val,iter0,label = 'Initial Guess = ' + str(0.9))
plt.plot(a_val,iter1,label = 'Initial Guess = ' + str(1.1))
plt.legend()
plt.grid(True)
plt.xlabel('a')
plt.ylabel('Iteration Count')
plt.title('Number of iteration in newton raphson method with increasing a')
plt.plot()

# %% [markdown]
# Result:
# 1. We notice, with increasing value of a , we can see the convergence towards the negative root of the function.
# 2. We also notice the relation between a and number of iterations to find the root are directly proportional.
