def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	w_new = []
	for params, grads, is_applied in zip(parameters, gradients, apply_to_all):
		new_params = []
		for w, grad in zip(params, grads):
			if (is_applied):
				w_update = w - lr * grad - lr * weight_decay * w
			else:
				w_update = w - lr * grad
			new_params.append(w_update)
		w_new.append(new_params)
	return w_new