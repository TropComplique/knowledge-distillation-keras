from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Flatten, Dense, Activation

def get_resnet():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    logits = Dense(256)(x)
    probabilities = Activation('softmax')(logits)
    return Model(base_model.input, probabilities)