def calculate_parameters(layers: list[dict]) -> int:
	total = 0
	for layer in layers:
		if layer['type'] == 'dense':
			total += layer['input_size'] * layer['output_size']
			if 'bias'not in layer or layer['bias']:
				total += layer['output_size']
		if layer['type'] == 'conv2d':
			kernel = layer['kernel_size'] ** 2
			inOut = layer['in_channels'] * layer['out_channels']
			total += kernel * inOut
			if 'bias' not in layer or layer['bias']:
				total += layer['out_channels'] 

	return total