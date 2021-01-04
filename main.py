import keras
from keras_position_wise_feed_forward import FeedForward
import keras2onnx

input_layer = keras.layers.Input(shape=(None, 32))
feed_forward_layer = FeedForward(units=128)(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=feed_forward_layer)
model.compile(optimizer='adam', loss='mse')
model.summary()

#keras.backend.set_learning_phase(0)

onnx_model = keras2onnx.convert_keras(model, 'feed_forward', debug_mode=1)
keras2onnx.save_model(onnx_model, 'foo.onnx')
