import pandas as pd 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 

train = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

train['room_per_person'] = train['total_rooms'] / train['population']
train['median_house_value'] = train['median_house_value'] / 1000.0

train[train.total_rooms < 1000]

lr = 0.001
epochs = 10
batch_size = 32
my_feature = 'total_rooms'
my_label = 'median_house_value'

def build_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=(1,))
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def train_model(model, df, feature, label, epochs, batch):

    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=None,
                        epochs=epochs)

    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[0]

    epochs = history.epoch 

    error_epoch = pd.DataFrame(history.history)

    rmse = error_epoch["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


def plot_model(trained_weight, trained_bias, feature, label):

    plt.xlabel(feature)
    plt.ylabel(label)

    random_example = train.sample(n=250)
    plt.scatter(random_example[feature], random_example[label])

    x_0 = 0
    y_0 = trained_bias
    x_1 = 10000
    y_1 = trained_bias[0] + (trained_weight[0]*x_1)

    plt.plot([x_0, x_1], [y_0, y_1], c='r')

    plt.show()


def plot_loss_curve(epochs, rmse):

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('rmse')

    plt.plot(epochs, rmse, label='loss')
    plt.legend()
    plt.show()


my_model = None

my_model = build_model(lr)
weight,bias,epoch,rmse = train_model(my_model, train, my_feature,
                                     my_label, epochs, batch_size)

print("\nLearned weight for model: {}".format(weight))
print("Learned bias for model: {}".format(bias))
print("")

plot_model(weight, bias, my_feature, my_label)
plot_loss_curve(epochs, rmse)


def predict_house_values(n, feature, label):

    batch = train[feature][10000:10000 + n]

    predicted_values = my_model.predict_on_batch(x=batch)

    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")

    for i in range(n):
        print("{}    {}    {}".format(train[feature][i],
                                    train[label][i]),
                                    predicted_values[i][0])

predict_house_values(15, my_feature, my_label)