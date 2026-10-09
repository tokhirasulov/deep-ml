import math

def activation_derivatives(x: float) -> dict[str, float]:

	sigmoid = 1 / (1 + math.exp(x))
	tanh = math.tanh(x)
	relu = max(0, x)

	dsigmoid = sigmoid * (1 - sigmoid)
	dtanh = 1 - tanh ** 2
	drelu = 0 if x <= 0 else 1
	return {
		'sigmoid': dsigmoid,
		'tanh': dtanh,
		'relu': drelu
	}

	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	pass