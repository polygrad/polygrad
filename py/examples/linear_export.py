"""Fit y = a*x+b and export for js/examples/linear_predict.js."""
import sys
from polygrad import Model, Tensor


class Linear:
    def __init__(self):
        self.a = Tensor([0.0])
        self.b = Tensor([0.0])

    def __call__(self, x):
        return {"prediction": self.a * x + self.b}


def main(path="linear.pgb"):
    model = Model(
        Linear(), inputs={"x": Tensor.empty(5)}, targets={"y": Tensor.empty(5)},
        loss=lambda outputs, y: (outputs["prediction"] - y).square().mean(),
    )
    try:
        model.fit({"x": [-2, -1, 0, 1, 2], "y": [-4, -1, 2, 5, 8]},
                  epochs=100, optimizer="sgd", lr=0.1)
        model.save(path, include_optimizer=False)
        print("a, b:", model.read_buffer("a"), model.read_buffer("b"))
    finally:
        model.dispose()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "linear.pgb")
