import numpy as np

def function(t: float, w: float):
    return t - w**2

def question_1():
    w = 1
    a, b = (0, 2)
    N = 10
    # set up h
    h = (b - a) / N

    for i in range(0, N):
        t = a + i * h
        # this gets the next approximation
        next_w = w + h * function(t, w)
        # we need to set the just solved "w" to be the original w
        w = next_w
        
        
    print("%.5f" % w)
    return None


def question_2():
    w = 1
    a, b = (0, 2)
    N = 10
    # set up h
    h = (b - a) / N

    for i in range(0, N):
        t = a + i * h
        # this gets the next approximation
        k1 = h * function(t, w)
        k2 = h * function(t + h/2, w + k1 / 2)
        k3 = h * function(t + h / 2, w + k2 / 2)
        k4 = h * function(t + h, w + k3)
        next_w = w + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # we need to set the just solved "w" to be the original w
        w = next_w

    print("%.5f" % w)
    return None

def question_3():

    Ab = [
        [ 2, -1,  1,  6],
        [ 1,  3,  1,  0],
        [-1,  5,  4, -3]
    ]
    new_Ab = np.copy(Ab)
    x = np.array([0, 0, 0], dtype=np.double)

    for i in range(0, 3):
        for j in range(0, 4):
            if(i == j):
                break
            for k in range(j, 4):
                new_Ab[i][k] = Ab[j][j] * Ab[i][k] - Ab[i][j] * Ab[j][k]
            Ab = np.copy(new_Ab)
            
    x[2] = (Ab[2][3]) / Ab[2][2]
    x[1] = (Ab[1][3] - Ab[1][2] * x[2]) / Ab[1][1]
    x[0] = (Ab[0][3] - Ab[0][1] * x[1] - Ab[0][2] * x[2]) / Ab[0][0]
    print(x)

    return None

def det(A, j):
    sum = 0
    size = np.size(A, 0)
    if(size == 1):
        return A[0][0]
    for i in range(0, size):
        M = np.delete(A, i, 0)
        M = np.delete(M, j, 1)
        sum += (-1)**(i + j) * A[i][j] * det(M, 0)
    return sum
    

def question_4():

    A = np.array([
        [ 1, 1, 0, 3],
        [ 2, 1,-1, 1],
        [ 3,-1,-1, 2],
        [-1, 2, 3,-1]
    ], dtype=np.double)
    
    determinant = det(A, 2)
    print("%.5f" % determinant)
    
    U = np.copy(A)
    L = np.zeros((4, 4))
    for i in range(0, 4):
        L[i][i] = 1
    for j in range(0, 4):
        for i in range(j, 4):
            if(i <= j or A[i][j] == 0):
                continue
            L[i, j] = A[i][j] * A[j][j]
            for k in range(j, 4):
                U[i][k] =  A[i][k] - A[i][j] * A[j][k] * A[j][j]
                # print(i, k, "=", A[j][j], "*", A[i][k], "-", A[i][j], "*", A[j][k], "=", U[i][k])
            A = np.copy(U)
            # print(A)

    print(L)
    print(U)
    
    return None

def question_5():

    A = np.array([
        [ 9, 0, 5, 2, 1],
        [ 3, 9, 1, 2, 1],
        [ 0, 1, 7, 2, 3],
        [ 4, 2, 3,12, 2],
        [ 3, 2, 4, 0, 8]
    ])
    for i in range(0, 5):
        for i in range(0, 5):
            if (np.absolute(A[i,i]) < np.sum(np.absolute(A[i])) - A[i,i]):
                print(False)
                return None
    print(True)
    return None

def question_6():

    A = np.array([
        [ 2, 2, 1],
        [ 2, 3, 0],
        [ 1, 0, 2]
    ])
    x = np.zeros((3, 3))
    for i in range(0, 3):
        x[i][i] = 1
    xT = x.transpose()

    if(np.array_equal(A, A.transpose()) == False):
        print(False)
        return None
    
    if(det(xT * A * x, 0) <= 0):
        print(False)
        return None
    
    print(True)
    return None

if __name__ == "__main__":
    question_1()
    question_2()
    question_3()
    question_4()
    question_5()
    question_6()
