import numpy as np
import tensorflow as tf


def generate_linear_data(a, b, count=100, noise=0.1):
    x = np.random.uniform(-10, 10, size=(count, 1)).astype(np.float32)
    y = (a * x + b + np.random.uniform(-noise, noise, size=(count, 1))).astype(
        np.float32
    )
    return x, y


def build_model():
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(1, input_shape=(1,), activation="linear")]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss="mse")
    return model


def main():
    print("TensorFlow neuron solving y = a*x + b")
    a = float(input("Enter coefficient a: "))
    b = float(input("Enter coefficient b: "))

    x_train, y_train = generate_linear_data(a, b, count=200, noise=0.2)
    model = build_model()
    model.fit(x_train, y_train, epochs=200, batch_size=16, verbose=0)

    weights, bias = model.layers[0].get_weights()
    print("\nTrained neuron parameters:")
    print(f"  weight = {weights[0][0]:.4f}")
    print(f"  bias   = {bias[0]:.4f}")

    for x_test in [-5, -1, 0, 1, 5]:
        predicted = model.predict(np.array([[x_test]], dtype=np.float32), verbose=0)[0, 0]
        actual = a * x_test + b
        print(f"x={x_test:>3} -> predicted={predicted:>7.3f}, actual={actual:>7.3f}")


if __name__ == "__main__":
    main()