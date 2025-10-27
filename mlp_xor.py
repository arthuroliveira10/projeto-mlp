import numpy as np

class MLP:
    def __init__(self):
        pass

    def train(self, inputs, outputs, alpha, epochs):
        w11, w12, w21, w22 = np.random.uniform(-1,1,4)
        wh1, wh2 = np.random.uniform(-1,1,2)
        b1, b2, b3 = np.random.uniform(-1,1,3)

        for i in range(epochs):
            for j in range(len(inputs)):
                h1 = 1 / (1 + np.exp(-((inputs[j][0]*w11) + (inputs[j][1]*w21) + b1)))
                h2 = 1 / (1 + np.exp(-((inputs[j][0]*w12) + (inputs[j][1]*w22) + b2)))
                y = 1 / (1 + np.exp(-((h1*wh1) + (h2*wh2) + b3)))
                error = outputs[j][0] - y

                derivative_y = y * (1 - y) * error
                derivative_h1 = h1 * (1 - h1) * wh1 * derivative_y
                derivative_h2 = h2 * (1 - h2) * wh2 * derivative_y

                w11 += alpha * derivative_h1 * inputs[j][0]
                w12 += alpha * derivative_h2 * inputs[j][0]
                w21 += alpha * derivative_h1 * inputs[j][1]
                w22 += alpha * derivative_h2 * inputs[j][1]
                wh1 += alpha * derivative_y * h1
                wh2 += alpha * derivative_y * h2
                b1 += alpha * derivative_h1
                b2 += alpha * derivative_h2
                b3 += alpha * derivative_y

        return w11, w12, w21, w22, wh1, wh2, b1, b2, b3

    def predict(self, weights, x1, x2):
        hidden1 = 1 / (1 + np.exp(-((x1*weights[0]) + (x2*weights[2]) + weights[6])))
        hidden2 = 1 / (1 + np.exp(-((x1*weights[1]) + (x2*weights[3]) + weights[7])))
        y = 1 / (1 + np.exp(-((hidden1*weights[4]) + (hidden2*weights[5]) + weights[8])))
        return 1 if y >= 0.5 else 0

if __name__ == "__main__":
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    outputs = [[0], [1], [1], [0]]
    mlp = MLP()
    trained_weights = mlp.train(inputs, outputs, alpha=0.05, epochs=50000)
    print("Resultados:")
    for e in inputs:
        print(f"{e} -> {mlp.predict(trained_weights, e[0], e[1])}")