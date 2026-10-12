def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	w_new = []
	for params, grads in zip(parameters, gradients):
		new_params = []
		for w, grad in zip(params, grads):
			w_update = w - lr * grad - lr * weight_decay * w
			new_params.append(w_update)
		w_new.append(new_params)
	return w_new