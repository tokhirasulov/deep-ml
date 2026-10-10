import math
def softplus(x: float) -> float:
	val = math.log(1 + math.exp(x))

	return round(val,4)