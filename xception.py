from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Activation, Dropout


def get_xception():
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    logits = Dense(256)(x)
    probabilities = Activation('softmax')(logits)
    return Model(base_model.input, probabilities)
