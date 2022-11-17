""" to process:

1.) Design Model(input, output size, forward pass)
2.) Contsruct loss and optimizer
3.) Training loop:
    - forward pass : compute prediciton
    - backward pass: gradients
    - update weights """ 
import torch
import torch.nn as nn


#Training 
lr = 0.01
n_iters = 10000

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) #[[1],[2],[3],[4]] because nn.Linear takes samples of feature  as input 

Y = torch.tensor([[2],[4],[6],[7]], dtype=torch.float32) #[[1],[2],[3],[4]] because nn.Linear takessamples of feature as input 


X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_fts = X.shape
#print(n_samples, n_fts)

""" we dont need w, because pytorch handels parameters 
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
"""

input_size = n_fts
output_size = n_fts

""" def forward(x):
        return w*x """

# instead of forward function, we can use pytorch nn.Linear
model = nn.Linear(input_size, output_size)

optimizer = torch.optim.SGD(model.parameters(), lr=lr) # SDG Optimizer
optimizer_2 = torch.optim.Adam(model.parameters(), lr=lr) #Adam Optimizer

#loss
loss = nn.MSELoss()

print(f"Learnrate = {lr}, Number of Iterations = {n_iters}\n")

print(f'Prediction before training : f(5) = {model(X_test).item():.3f}')

#Training 
for epoch in range(n_iters):
    #prediciton = forward pass
    y_pred = model(X)
    #loss
    l = loss(Y, y_pred)

    #graident = backward pass
    l.backward()

	#update weights wtich opt.step function, optizmier handels weights
    optimizer.step()
   # optimizer_2.step()

    #empty gradients
    optimizer.zero_grad()
   # optimizer_2.zero_grad()

    if epoch % 100 == 0:
        [w,b] = model.parameters()
        print(f"epoch {epoch+1} : w = {w[0][0].item():.3f}, loss = {l:.3f}")

#computainal gradient not as exact as numerical, therefore more iterations needed
print(f"prediticon after trainig: f(5) : {model(X_test).item()}")
