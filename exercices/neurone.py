import numpy as np
import tensorflow as tf


def generate_linear_data(a, b, n=300, noise=0.3):
    x = np.random.uniform(-10, 10, size=(n, 1)).astype(np.float32)
    y = (a * x + b + np.random.uniform(-noise, noise, size=(n, 1))).astype(
        np.float32
    )
    return x, y


def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(8, activation="relu", input_shape=(1,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main():
    print("TensorFlow network approximating y = a*x + b")
    a = float(input("Enter coefficient a: "))
    b = float(input("Enter coefficient b: "))

    x_train, y_train = generate_linear_data(a, b, n=300, noise=0.2)
    model = build_model()
    model.fit(x_train, y_train, epochs=250, batch_size=16, verbose=0)

    print("\nTrained model layer shapes:")
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            w, bias = layer.get_weights()
            print(f"  {layer.name}: weights {w.shape}, bias {bias.shape}")

    for x_test in [-8.0, -2.0, 0.0, 2.0, 8.0]:
        predicted = model.predict(
            np.array([[x_test]], dtype=np.float32), verbose=0)[0, 0]
        actual = a * x_test + b
        print(f"x={x_test:>4} -> predicted={predicted:>7.3f}, actual={actual:>7.3f}")


if __name__ == "__main__":
    main()