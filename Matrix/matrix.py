from tensor import Tensor

class Matrix(Tensor):
    def __init__(self, dim: tuple, data: list):
        if not(isinstance(dim, tuple) and len(dim) == 2 and dim[0] * dim[1] == len(data)):
            raise ValueError("Wrong dimension")
        super().__init__(dim, data)
        self.rows, self.cols = dim

    def conv_rc2i(self, row_idx: int, col_idx: int):
        if not(0 <= row_idx < self.rows and 0 <= col_idx < self.cols):
            raise IndexError("Wrong index")
        return row_idx * self.cols + col_idx

    def conv_i2rc(self, idx: int):
        if not(0 <= idx < len(self.data)):
            raise IndexError("Wrong index")
        return idx // self.cols, idx % self.cols

    def __str__(self):
        res = "[\n"
        max_len = max([len(str(value)) for value in self.data])
        for row_idx in range(self.rows):
            for col_idx in range(self.cols):
                elem_idx = self.conv_rc2i(row_idx, col_idx)
                res += ' ' * (max_len - len(str(self.data[elem_idx])) + 2) + str(self.data[elem_idx])
            res += '\n\n'
        res = res[:-1] + ']'
        return res

    def __getitem__(self, key: int | list | slice | tuple):
        if isinstance(key, int):
            if key < -self.rows or key >= self.rows:
                raise IndexError("Row index is out of range")
            key = key % self.rows
            return Matrix((1, self.cols), self.data[key * self.cols : (key + 1) * self.cols])

        if isinstance(key, list):
            if all(isinstance(k, int) for k in key):
                data = [self.data[r * self.cols:(r + 1) * self.cols] for r in key]
                return Matrix((len(key), self.cols), sum(data, []))

        if isinstance(key, slice):
            row_indices = range(*key.indices(self.rows))
            data = [self.data[r * self.cols : (r + 1) * self.cols] for r in row_indices]
            return Matrix((len(row_indices), self.cols), [elem for row in data for elem in row])

        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Tuple must have two elements")

            row_key, col_key = key
            if isinstance(row_key, int) and isinstance(col_key, int):
                return self.data[self.conv_rc2i(row_key % self.rows, col_key % self.cols)]

            if isinstance(row_key, int):
                row_indices = [row_key]
            elif isinstance(row_key, slice):
                row_indices = range(*row_key.indices(self.rows))
            elif isinstance(row_key, list):
                row_indices = row_key
            else:
                raise TypeError("Invalid row index type")

            if isinstance(col_key, int):
                col_indices = [col_key]
            elif isinstance(col_key, slice):
                col_indices = range(*col_key.indices(self.cols))
            elif isinstance(col_key, list):
                col_indices = col_key
            else:
                raise TypeError("Invalid column index type")

            data = [self.data[self.conv_rc2i(r, c)] for r in row_indices for c in col_indices]
            return Matrix((len(row_indices), len(col_indices)), data)

        raise TypeError("Invalid index type")