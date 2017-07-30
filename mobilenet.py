from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Activation, GlobalAveragePooling2D, Dropout, Dense

def get_mobilenet():
    base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(1e-3)(x)
    logits = Dense(256)(x)
    probabilities = Activation('softmax')(logits)
    return Model(base_model.input, probabilities)