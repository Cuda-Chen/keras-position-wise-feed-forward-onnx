import keras
from keras_position_wise_feed_forward import FeedForward

input_layer = keras.layers.Input(shape=(None, 32))
feed_forward_layer = FeedForward(units=128)(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=feed_forward_layer)
model.compile(optimizer='adam', loss='mse')
model.summary()
