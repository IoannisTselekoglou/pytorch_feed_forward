import torch
import numpy as np




#Linear Regression :

# f = w*x ; w = weights
# f = 2*x 



X = np.array([1,2,3,4], dtype=np.float32)

Y = np.array([2,4,6,8], dtype=np.float32)

# model prediciton

w = 0.0

def forward(x):
    return w*x

#loss

def loss(y, y_pred):
    #return abs(y_pred-y)
    return ((y_pred-y)**2).mean()


#gradient
#MSE = dj = 1/N * (w*x -y)**2
#dJ/dw = 1/N * 2x*(w*x -y); ableitung

def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred - y).mean()

print(f'Prediction before training : f(5) = {forward(5):.3f}')

#Training 
lr = 0.01
n_iters = 20

for epoch in range(n_iters):
    #prediciton = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #graident
    dw = gradient(X,Y,y_pred)

    #update weights
    w -= dw*lr

    if epoch % 3 == 0:
        print(f"epoch {epoch+1} : w = {w:.3f}, loss = {l:.3f}")
        print(dw)


print(f"prediticon after trainig: f(5) : {forward(5)}")




