import keras
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Activation, GlobalAveragePooling2D, Dropout, Dense, Input


def get_mobilenet():
    base_model = MobileNet(
        include_top=False, weights='imagenet', 
        input_tensor=Input(shape=(299, 299, 3))
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    logits = Dense(256, kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    probabilities = Activation('softmax')(logits)
    return Model(base_model.input, probabilities)
