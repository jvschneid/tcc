import torch

dtype = torch.cuda.FloatTensor

batch_size, d_in, hidden_dim, d_out = 64, 1000, 100, 10

# Manual Version
x = torch.randn(batch_size, d_in).type(dtype)
y = torch.randn(batch_size, d_out).type(dtype)

w1 = torch.randn(d_in, hidden_dim).type(dtype)
w2 = torch.randn(hidden_dim, d_out).type(dtype)

lr = 1e-6

for t in range(500):
	h = x.mm(w1)
	h_relu = h.clamp(min=0)
	y_pred = h_relu.mm(w2)

	loss = (y_pred - y).pow(2).sum()
	print(t, loss)

	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	grad_h = grad_h_relu.clone()
	grad_h[h < 0] = 0
	grad_w1 = x.t().mm(grad_h)

	w1 -= lr * grad_w1
	w2 -= lr * grad_w2

# Autograd Version
from torch.autograd import Variable

x = Variable(torch.randn(batch_size, d_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(batch_size, d_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(d_in, hidden_dim).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(hidden_dim, d_out).type(dtype), requires_grad=True)

for t in range(500):
	y_pred = x.mm(w1).clamp(min=0).mm(w2)

	loss = (y_pred - y).pow(2).sum()
	print(t, loss.data[0])

	loss.backward()

	w1.data -= lr * w1.grad.data
	w2.data -= lr * w2.grad.data

	w1.grad.data.zero_()
	w2.grad.data.zero_()

# Custom autograd Function

class MyReLU(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		return input.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input < 0] = 0
		return grad_input

x = Variable(torch.randn(batch_size, d_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(batch_size, d_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(d_in, hidden_dim).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(hidden_dim, d_out).type(dtype), requires_grad=True)

for t in range(500):
	relu = MyReLU.apply

	y_pred = relu(x.mm(w1)).mm(w2)

	loss = (y_pred - y).pow(2).sum()
	print(t, loss.data[0])

	loss.backward()

	w1.data -= lr * w1.grad.data
	w2.data -= lr * w2.grad.data

	w1.grad.data.zero_()
	w2.grad.data.zero_()