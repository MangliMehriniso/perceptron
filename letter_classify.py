import numpy as np

class FontSize:
    Small = 0
    Medium = 1
    Large = 2

def printMatrix(matrix):
    for e in matrix:
        print e

def printMatrix_v2(matrix):
    for i in range(len(matrix)):
        s = ""
        for j in range(len(matrix[i])):
            if matrix[i][j] == -1:
                s += '{:2}'.format('.')
            else:
                s += '{:2}'.format('#')
        print s

def createMatrix(w, h, defaultValue):
    return [[defaultValue for x in range(w)] for y in range(h)]

def creatLetterA(size):
    result = createMatrix(7,9,-1)
    if size == FontSize.Small:
        result[0][2] = 1
        result[0][3] = 1
        result[1][3] = 1
        result[2][3] = 1
        result[3][2] = 1
        result[3][4] = 1
        result[4][2] = 1
        result[4][4] = 1
        result[5][1] = 1
        result[5][2] = 1
        result[5][3] = 1
        result[5][4] = 1
        result[5][5] = 1
        result[6][1] = 1
        result[6][5] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][0] = 1
        result[8][1] = 1
        result[8][2] = 1
        result[8][4] = 1
        result[8][5] = 1
        result[8][6] = 1
    elif size == FontSize.Medium:
        result[0][3] = 1
        result[1][3] = 1
        result[2][3] = 1
        result[3][2] = 1
        result[3][4] = 1
        result[4][2] = 1
        result[4][4] = 1
        result[5][1] = 1
        result[5][5] = 1
        result[6][1] = 1
        result[6][2] = 1
        result[6][3] = 1
        result[6][4] = 1
        result[6][5] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][1] = 1
        result[8][5] = 1
    elif size == FontSize.Large:
        result[0][3] = 1
        result[1][3] = 1
        result[2][2] = 1
        result[2][4] = 1
        result[3][2] = 1
        result[3][4] = 1
        result[4][1] = 1
        result[4][5] = 1
        result[5][1] = 1
        result[5][2] = 1
        result[5][3] = 1
        result[5][4] = 1
        result[5][5] = 1
        result[6][0] = 1
        result[6][6] = 1
        result[7][0] = 1
        result[7][6] = 1
        result[8][0] = 1
        result[8][1] = 1
        result[8][5] = 1
        result[8][6] = 1
    return result

def creatLetterB(size):
    result = createMatrix(7,9,-1)
    if size == FontSize.Small:
        result[0][0] = 1    
        result[0][1] = 1    
        result[0][2] = 1    
        result[0][3] = 1    
        result[0][4] = 1    
        result[0][5] = 1    
        result[1][1] = 1    
        result[1][6] = 1    
        result[2][1] = 1    
        result[2][6] = 1    
        result[3][1] = 1    
        result[3][6] = 1    
        result[4][1] = 1    
        result[4][2] = 1    
        result[4][3] = 1    
        result[4][4] = 1    
        result[4][5] = 1    
        result[5][1] = 1    
        result[5][6] = 1    
        result[6][1] = 1    
        result[6][6] = 1    
        result[7][1] = 1    
        result[7][6] = 1    
        result[8][0] = 1    
        result[8][1] = 1    
        result[8][2] = 1    
        result[8][3] = 1    
        result[8][4] = 1    
        result[8][5] = 1    
    elif size == FontSize.Medium:
        result[0][0] = 1    
        result[0][1] = 1    
        result[0][2] = 1    
        result[0][3] = 1    
        result[0][4] = 1    
        result[0][5] = 1    
        result[1][0] = 1    
        result[1][6] = 1    
        result[2][0] = 1    
        result[2][6] = 1    
        result[3][0] = 1    
        result[3][6] = 1    
        result[4][0] = 1    
        result[4][1] = 1    
        result[4][2] = 1    
        result[4][3] = 1    
        result[4][4] = 1    
        result[4][5] = 1    
        result[5][0] = 1    
        result[5][6] = 1    
        result[6][0] = 1    
        result[6][6] = 1    
        result[7][0] = 1    
        result[7][6] = 1    
        result[8][0] = 1    
        result[8][0] = 1    
        result[8][1] = 1    
        result[8][2] = 1    
        result[8][3] = 1    
        result[8][4] = 1    
        result[8][5] = 1
    elif size == FontSize.Large:
        result[0][0] = 1    
        result[0][1] = 1    
        result[0][2] = 1    
        result[0][3] = 1    
        result[0][4] = 1    
        result[0][5] = 1    
        result[1][1] = 1    
        result[1][6] = 1    
        result[2][1] = 1    
        result[2][6] = 1    
        result[3][1] = 1    
        result[3][2] = 1    
        result[3][3] = 1    
        result[3][4] = 1    
        result[3][5] = 1

        result[4][1] = 1    
        result[4][6] = 1    


        result[5][1] = 1    
        result[5][6] = 1    
        result[6][1] = 1    
        result[6][6] = 1    
        result[7][1] = 1    
        result[7][6] = 1    
        result[8][0] = 1    
        result[8][1] = 1    
        result[8][2] = 1    
        result[8][3] = 1    
        result[8][4] = 1    
        result[8][5] = 1    
    return result

def createLetterC(size):
    result = createMatrix(7,9,-1)
    if (size == FontSize.Small):
        result[0][2] = 1
        result[0][3] = 1
        result[0][4] = 1
        result[0][5] = 1
        result[0][6] = 1
        result[1][1] = 1
        result[1][6] = 1
        result[2][0] = 1
        result[3][0] = 1
        result[4][0] = 1
        result[5][0] = 1
        result[6][0] = 1
        result[7][1] = 1
        result[7][6] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
        result[8][5] = 1
    elif (size == FontSize.Medium):
        result[0][2] = 1
        result[0][3] = 1
        result[0][4] = 1
        result[1][1] = 1
        result[1][5] = 1
        result[2][0] = 1
        result[2][6] = 1
        result[3][0] = 1
        result[4][0] = 1
        result[5][0] = 1
        result[6][0] = 1
        result[6][6] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
    elif (size == FontSize.Large):
        result[0][2] = 1
        result[0][3] = 1
        result[0][4] = 1
        result[0][6] = 1
        result[1][1] = 1
        result[1][5] = 1
        result[1][6] = 1
        result[2][0] = 1
        result[2][6] = 1
        result[3][0] = 1
        result[4][0] = 1
        result[5][0] = 1
        result[6][0] = 1
        result[6][6] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
    return result

def createLetterD(size):
    result = createMatrix(7,9,-1)
    if size == FontSize.Small:
        result[0][0] = 1
        result[0][1] = 1
        result[0][2] = 1
        result[0][3] = 1
        result[0][4] = 1
        result[1][1] = 1
        result[1][5] = 1
        result[2][1] = 1
        result[2][6] = 1
        result[3][1] = 1
        result[3][6] = 1
        result[4][1] = 1
        result[4][6] = 1
        result[5][1] = 1
        result[5][6] = 1
        result[6][1] = 1
        result[6][6] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][0] = 1
        result[8][1] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
    elif size == FontSize.Medium:
        result[0][0] = 1
        result[0][1] = 1
        result[0][2] = 1
        result[0][3] = 1
        result[0][4] = 1
        result[1][0] = 1
        result[1][5] = 1
        result[2][0] = 1
        result[2][6] = 1
        result[3][0] = 1
        result[3][6] = 1
        result[4][0] = 1
        result[4][6] = 1
        result[5][0] = 1
        result[5][6] = 1
        result[6][0] = 1
        result[6][6] = 1
        result[7][0] = 1
        result[7][5] = 1
        result[8][0] = 1
        result[8][1] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
    elif size == FontSize.Large:
        result[0][0] = 1
        result[0][1] = 1
        result[0][2] = 1
        result[0][3] = 1
        result[0][4] = 1
        result[1][1] = 1
        result[1][5] = 1
        result[2][1] = 1
        result[2][6] = 1
        result[3][1] = 1
        result[3][6] = 1
        result[4][1] = 1
        result[4][6] = 1
        result[5][1] = 1
        result[5][6] = 1
        result[6][1] = 1
        result[6][6] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][0] = 1
        result[8][1] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
    return result

def createLetterE(size):
    result = createMatrix(7,9,-1)
    if size == FontSize.Small:
        result[0][0] = 1
        result[0][1] = 1
        result[0][2] = 1
        result[0][3] = 1
        result[0][4] = 1
        result[0][5] = 1
        result[0][6] = 1
        result[1][1] = 1
        result[1][6] = 1
        result[2][1] = 1
        result[3][1] = 1
        result[3][3] = 1
        result[4][1] = 1
        result[4][2] = 1
        result[4][3] = 1
        result[5][1] = 1
        result[5][3] = 1
        result[6][1] = 1
        result[7][1] = 1
        result[7][6] = 1
        result[8][0] = 1
        result[8][1] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
        result[8][5] = 1
        result[8][6] = 1
    elif size == FontSize.Medium:
        result[0][0] = 1
        result[0][1] = 1
        result[0][2] = 1
        result[0][3] = 1
        result[0][4] = 1
        result[0][5] = 1
        result[0][6] = 1
        result[1][0] = 1
        result[2][0] = 1
        result[3][0] = 1
        result[4][0] = 1
        result[4][1] = 1
        result[4][2] = 1
        result[4][3] = 1
        result[4][4] = 1
        result[5][0] = 1
        result[6][0] = 1
        result[7][0] = 1
        result[8][0] = 1
        result[8][1] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
        result[8][5] = 1
        result[8][6] = 1
    elif size == FontSize.Large:
        result[0][0] = 1
        result[0][1] = 1
        result[0][2] = 1
        result[0][3] = 1
        result[0][4] = 1
        result[0][5] = 1
        result[0][6] = 1
        result[1][1] = 1
        result[1][6] = 1
        result[2][1] = 1
        result[2][4] = 1
        result[3][1] = 1
        result[3][2] = 1
        result[3][3] = 1
        result[3][4] = 1
        result[4][1] = 1
        result[4][4] = 1
        result[5][1] = 1
        result[6][1] = 1
        result[7][1] = 1
        result[7][6] = 1
        result[8][0] = 1
        result[8][1] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
        result[8][5] = 1
        result[8][6] = 1
    return result

def createLetterJ(size):
    result = createMatrix(7,9,-1)
    if size == FontSize.Small:
        result[0][3] = 1
        result[0][4] = 1
        result[0][5] = 1
        result[0][6] = 1
        result[1][5] = 1
        result[2][5] = 1
        result[3][5] = 1
        result[4][5] = 1
        result[5][5] = 1
        result[6][1] = 1
        result[6][5] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
    elif size == FontSize.Medium:
        result[0][6] = 1
        result[2][5] = 1
        result[3][5] = 1
        result[4][5] = 1
        result[5][5] = 1
        result[6][1] = 1
        result[6][5] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
    elif size == FontSize.Large:
        result[0][4] = 1
        result[0][5] = 1
        result[0][6] = 1
        result[1][5] = 1
        result[2][5] = 1
        result[3][5] = 1
        result[4][5] = 1
        result[5][5] = 1
        result[6][5] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][2] = 1
        result[8][3] = 1
        result[8][4] = 1
    return result

def createLetterK(size):
    result = createMatrix(7,9,-1)
    if (size == FontSize.Small):
        result[0][0] = 1
        result[0][1] = 1
        result[0][2] = 1
        result[0][5] = 1
        result[0][6] = 1
        result[1][1] = 1
        result[1][4] = 1
        result[2][1] = 1
        result[2][3] = 1
        result[3][1] = 1
        result[3][2] = 1
        result[4][1] = 1
        result[4][2] = 1
        result[5][1] = 1
        result[5][3] = 1
        result[6][1] = 1
        result[6][4] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][0] = 1
        result[8][1] = 1
        result[8][2] = 1
        result[8][5] = 1
        result[8][6] = 1
    elif (size == FontSize.Medium):
        result[0][0] = 1
        result[0][5] = 1
        result[1][0] = 1
        result[1][4] = 1
        result[2][0] = 1
        result[2][3] = 1
        result[3][0] = 1
        result[3][2] = 1
        result[4][0] = 1
        result[4][1] = 1
        result[5][0] = 1
        result[5][2] = 1
        result[6][0] = 1
        result[6][3] = 1
        result[7][0] = 1
        result[7][4] = 1
        result[8][0] = 1
        result[8][5] = 1
    elif (size == FontSize.Large):
        result[0][0] = 1
        result[0][1] = 1
        result[0][2] = 1
        result[0][5] = 1
        result[0][6] = 1
        result[1][1] = 1
        result[1][5] = 1
        result[2][1] = 1
        result[2][4] = 1
        result[3][1] = 1
        result[3][3] = 1
        result[4][1] = 1
        result[4][2] = 1
        result[5][1] = 1
        result[5][3] = 1
        result[6][1] = 1
        result[6][4] = 1
        result[7][1] = 1
        result[7][5] = 1
        result[8][0] = 1
        result[8][1] = 1
        result[8][2] = 1
        result[8][5] = 1
        result[8][6] = 1
    return result



def train(s, t,threshold=0,learning_rate=1):
    weights = createMatrix(63, 7, 0)
    biases=np.zeros(7)
    net = np.zeros(7)
    stop=False
    while(stop==False):
        y=np.full(7,-1)
        for j in range(len(t)):
            net[j]=biases[j]+sum([m*n for m,n in zip(weights[j],s)])
            if net[j]>threshold:
                y[j]=1
            elif net[j]>=-threshold and net[j]<=threshold:
                y[j]=0
            elif net[j]<threshold:
                y[j]=-1
        for i in range(len(t)):
            for k in range(len(s)):
                if (t[i]!=y[i]):
                    weights[i][k]+=learning_rate*t[i]*s[k]
                    biases[i] += learning_rate * t[i]
                else:
                    stop=True
    return weights,biases

if __name__ == "__main__":
    A_Small = creatLetterA(FontSize.Small)
    A_Medium = creatLetterA(FontSize.Medium)
    A_Large = creatLetterA(FontSize.Large)

    B_Small = creatLetterB(FontSize.Small)
    B_Medium = creatLetterB(FontSize.Medium)
    B_Large = creatLetterB(FontSize.Large)

    C_Small = createLetterC(FontSize.Small)
    C_Medium = createLetterC(FontSize.Medium)
    C_Large = createLetterC(FontSize.Large)

    D_Small = createLetterD(FontSize.Small)
    D_Medium = createLetterD(FontSize.Medium)
    D_Large = createLetterD(FontSize.Large)

    E_Small = createLetterE(FontSize.Small)
    E_Medium = createLetterE(FontSize.Medium)
    E_Large = createLetterE(FontSize.Large)

    J_Small = createLetterJ(FontSize.Small)
    J_Medium = createLetterJ(FontSize.Medium)
    J_Large = createLetterJ(FontSize.Large)

    K_Small = createLetterK(FontSize.Small)
    K_Medium = createLetterK(FontSize.Medium)
    K_Large = createLetterK(FontSize.Large)

    printMatrix_v2(K_Medium)
    s=[]
    t=[]
    s.append(np.matrix(A_Small).getA1())
    s.append(np.matrix(B_Small).getA1())
    s.append(np.matrix(C_Small).getA1())
    s.append(np.matrix(D_Small).getA1())
    s.append(np.matrix(E_Small).getA1())
    s.append(np.matrix(J_Small).getA1())
    s.append(np.matrix(K_Small).getA1())

    t.append([1, -1, -1, -1, -1, -1, -1])
    t.append([-1, 1, -1, -1, -1, -1, -1])
    t.append([-1, -1, 1, -1, -1, -1, -1])
    t.append([-1, -1, -1, 1, -1, -1, -1])
    t.append([-1, -1, -1, -1, 1, -1, -1])
    t.append([-1, -1, -1, -1, -1, 1, -1])
    t.append([-1, -1, -1, -1, -1, -1, 1])

    weight=[]#weights for each class
    bias=[]#biases for each class
    for i in range(len(t)):
        w,b=train(s[i],t[i])#train for each pair
        weight.append(w)
        bias.append(b)

    # y=np.zeros(7)
    #     # s=np.matrix(K_Small).getA1()
    #     # for i in range(7):
    #     #     y[i]=bias[i][i]+sum([m*n for m,n in zip(weight[i][i],s)])