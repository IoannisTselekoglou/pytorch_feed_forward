import torch
import numpy as np


#instead of manual gradient calc we use pytorch

#Linear Regression :

# f = w*x ; w = weights
# f = 2*x 



X = torch.tensor([1,2,3,4], dtype=torch.float32)

Y = torch.tensor([2,4,6,8], dtype=torch.float32)

# model prediciton

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w*x

#loss

def loss(y, y_pred):
    #return abs(y_pred-y)
    return ((y_pred-y)**2).mean()


#gradient
#MSE = dj = 1/N * (w*x -y)**2
#dJ/dw = 1/N * 2x*(w*x -y); ableitung

print(f'Prediction before training : f(5) = {forward(5):.3f}')

#Training 
lr = 0.01
n_iters = 100

for epoch in range(n_iters):
    #prediciton = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #graident = backward pass
    #dw = gradient(X,Y,y_pred)
    l.backward()

    #update weights; no_grad because weights should not have grad
    with torch.no_grad():
         w -= w.grad*lr
    #empty gradients
    w.grad.zero_()


    if epoch % 10 == 0:
        print(f"epoch {epoch+1} : w = {w:.3f}, loss = {l:.3f}")
        print(w)

#computainal gradient not as exact as numerical, therefore more iterations needed
print(f"prediticon after trainig: f(5) : {forward(5)}")

