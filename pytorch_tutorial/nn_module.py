import torch
from torch.autograd import Variable

batch_size, d_in, hidden_dim, d_out = 64, 1000, 100, 10

x = Variable(torch.randn(batch_size, d_in))
y = Variable(torch.randn(batch_size, d_out), requires_grad=False)

model = torch.nn.Sequential(
	torch.nn.Linear(d_in, hidden_dim),
	torch.nn.ReLU(),
	torch.nn.Linear(hidden_dim, d_out)
)

loss_fn = torch.nn.MSELoss(size_average=False)

lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for t in range(500):
	y_pred = model(x)

	loss = loss_fn(y_pred, y)
	print(t, loss.data[0])

	model.zero_grad()

	loss.backward()

	# Manual Update
	"""
	for param in model.parameters():
		param.data -= lr * param.grad.data
	"""

	# Automatic Update
	optimizer.step()

# Custom Module

class TwoLayerNet(torch.nn.Module):
	def __init__(self, d_in, hidden_dim, d_out):
		super(TwoLayerNet, self).__init__()
		self.linear1 = torch.nn.Linear(d_in, hidden_dim)
		self.linear2 = torch.nn.Linear(hidden_dim, d_out)

	def forward(self, x):
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		return y_pred

x = Variable(torch.randn(batch_size, d_in))
y = Variable(torch.randn(batch_size, d_out), requires_grad=False)

model = TwoLayerNet(d_in, hidden_dim, d_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for t in range(500):
	y_pred = model(x)

	loss = criterion(y_pred, y)
	print(t, loss.data[0])

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()