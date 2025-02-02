import math


class Op:
    def __init__(self, *, method, result=None, operand1: "Value", operand2: "Value"):
        self.method = method
        self.result = result
        self.operand1 = operand1
        self.operand2 = operand2


class Value:
    def __init__(self, value, tree=None):
        self.value = value
        self.grad = 0
        self.grad_func = []

        if tree is None:
            self.tree = Op(
                **{"method": "value", "operand1": self.value, "operand2": None}
            )
        else:
            self.tree = tree

    @staticmethod
    def add_to_tree(method, result, operand1, operand2):
        return Op(
            **{
                "method": method,
                "result": result,
                "operand1": operand1,
                "operand2": operand2,
            }
        )

    def __repr__(self):
        return f"Value(value={self.value}, grad={self.grad})"

    def _apply_operation(self, other, operation, method_name):
        if not isinstance(other, Value):
            other = Value(other)
        new_value = operation(self.value, other.value)
        new_tree = self.add_to_tree(method_name, new_value, self, other)
        return Value(new_value, new_tree)

    def __add__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x + y, "+")

    def __radd__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x + y, "+")

    def __iadd__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x + y, "+")

    def __sub__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x - y, "-")

    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other._apply_operation(self, lambda x, y: x - y, "-")

    def __mul__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x * y, "*")

    def __pow__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x**y, "**")

    def __truediv__(self, other: "Value"):
        return self * (other**-1)

    def __rtruediv__(self, other: "Value"):
        return other * (self**-1)

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):
        return self * other

    def relu(self):
        return self._apply_operation(None, lambda x, _: 0 if x < 0 else x, "ReLU")

    def backward(self):
        if not isinstance(self.tree, Op):
            raise ValueError("Operation tree is corrupted.")

        self.grad = 1

        topo_order = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if isinstance(v.tree, Op) and not v.tree.method == "value":
                    if v.tree.operand1:
                        build_topo(v.tree.operand1)
                    if v.tree.operand2:
                        build_topo(v.tree.operand2)
                topo_order.append(v)

        build_topo(self)

        for v in reversed(topo_order):
            if not isinstance(v.tree, Op):
                continue

            operand1 = v.tree.operand1
            operand2 = v.tree.operand2
            method = v.tree.method

            if method == "+":
                operand1.grad += v.grad * 1
                operand2.grad += v.grad * 1

            if method == "-":
                operand1.grad += v.grad * 1
                operand2.grad += v.grad * -1

            if method == "**":
                operand1.grad += (
                    v.grad * operand2.value * (operand1.value ** (operand2.value - 1))
                )

                try:
                    operand2.grad += (
                        v.grad
                        * operand1.value**operand2.value
                        * math.log(operand1.value)
                    )
                except ValueError:
                    pass

            if method in ["*"]:
                operand1.grad += v.grad * operand2.value
                operand2.grad += v.grad * operand1.value

            if method == "ReLU":
                operand1.grad += v.grad * (0 if operand1.value <= 0 else 1)

            if method == "value":
                pass


#  If you have diverging paths, you just add the gradients together
# If you have diverging paths, the grad_funcs will overwrite, hence use a grad_func queue,
#  or just apply the grad func immediately
