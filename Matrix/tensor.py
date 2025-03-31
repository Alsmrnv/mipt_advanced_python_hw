class Tensor:
    def __init__(self, dim: int | tuple, data: list):
        if not(isinstance(dim, int) or isinstance(dim, tuple)):
            raise ValueError("Wrong dimension. Dimension must be either int value or tuple of int values")
        if isinstance(dim, tuple) and not(all(isinstance(x, int) and x > 0 for x in dim)):
            raise ValueError("Wrong dimension. Dimension must be either int value or tuple of positive int values")
        if not(isinstance(data, list)):
            raise ValueError("Tensor's data must be list of values\n")
        self.dimension = dim
        self.data = data

    def __repr__(self):
        return str(self.data)