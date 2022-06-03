import datetime
import json
import os.path

import keras
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from src.utils.misc import make_directory


def load_data(json_path: str):
    with open(json_path, "r") as fp:
        data = json.load(fp)
        X = np.array(data["mfcc"])
        y = np.array(data["labels"])

        print(X.shape)
        print(y.shape)

        return X, y


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    validation_data = (X_test, y_test)
    train_data = (X_train, y_train)
    print("Training set: ")
    print(X_train.shape)
    print(y_train.shape)

    print("Testing set: ")
    print(X_test.shape)
    print(y_test.shape)

    return train_data, validation_data


def compile_and_train(model, train_data, validation_data, name, training_dump_path: str, lr=0.0001):
    make_directory(path=training_dump_path)
    optimiser = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model_path = training_dump_path + f"saved_model_{name}.h5"
    if os.path.exists(model_path):
        saved_model = load_model(filepath=model_path)
        # TODO: Extract history from saved model (or remove this part)

    # Train the model
    X_train, y_train = train_data
    print("Start time:", datetime.datetime.now().time())
    print("Training in progress...")
    history = model.fit(X_train, y_train, validation_data=validation_data, batch_size=32, epochs=50, verbose=1)
    print("Training finished")
    print("End time: ", datetime.datetime.now().time())
    model.save(filepath=training_dump_path + f"saved_model_{name}.h5")

    print("\nTraining accuracy: ", history.history["accuracy"][-1])
    print("Validation accuracy: ", history.history["val_accuracy"][-1])
    print("Training error (loss): ", history.history["loss"][-1])
    print("Validation error (loss): ", history.history["val_loss"][-1])

    # Save training history in JSON file
    with open(training_dump_path + "training_history_" + name + ".json", 'w') as file:
        json.dump(history.history, file)

    return history


def plot_results(history, name, out_path: str):
    make_directory(path=out_path)
    fig, axs = plt.subplots(2)

    # Accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # Error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.tight_layout()
    plt.savefig(out_path + "train_test_plot_" + name + ".png", format="png", dpi=200)
    plt.show()


if __name__ == "__main__":
    # Load MFCC dataset
    X, y = load_data(json_path="./../../data/gtzan_mfcc_json.json")
    train_data, validation_data = split_dataset(X=X, y=y)

    # Prepare model
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # Train model
    history = compile_and_train(model, train_data=train_data, validation_data=validation_data, name="basic",
                                training_dump_path="./../../model/")

    # Plot training results
    plot_results(history=history, name="basic", out_path="./../../img/")
