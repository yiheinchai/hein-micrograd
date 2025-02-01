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

    def _apply_operation(self, other, operation, method_name):
        if not isinstance(other, Value):
            other = Value(other)
        new_value = operation(self.value, other.value)
        new_tree = self.add_to_tree(method_name, new_value, self, other)
        return Value(new_value, new_tree)

    def __add__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x + y, "add")

    def __radd__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x + y, "add")

    def __iadd__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x + y, "add")

    def __sub__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x - y, "sub")

    def __mul__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x * y, "mul")

    def __pow__(self, other: "Value"):
        return self._apply_operation(other, lambda x, y: x**y, "pow")

    def __truediv__(self, other: "Value"):
        return self * (other**-1)

    def __neg__(self):
        return self * -1

    def backward(self):
        self._backward(root=True)

    def _backward(self, root=False):
        if not isinstance(self.tree, Op):
            raise ValueError("Operation tree is corrupted.")

        operand1 = self.tree.operand1
        operand2 = self.tree.operand2

        if root:
            self.grad = 1

        method = self.tree.method
        if method in ["add", "sub"]:
            operand1.grad += self.grad * 1
            operand2.grad += self.grad * 1

            operand1._backward()
            operand2._backward()

        if method == "pow":
            operand1.grad += (
                self.grad * operand2.value * (operand1.value ** (operand2.value - 1))
            )
            operand2.grad += (
                self.grad * operand1.value**operand2.value * math.log(operand1.value)
            )

            operand1._backward()
            operand2._backward()

        if method in ["mul"]:
            operand1.grad += self.grad * operand2.value
            operand2.grad += self.grad * operand1.value

            operand1._backward()
            operand2._backward()

        if method == "value":
            pass


#  If you have diverging paths, you just add the gradients together
# If you have diverging paths, the grad_funcs will overwrite, hence use a grad_func queue,
#  or just apply the grad func immediately
