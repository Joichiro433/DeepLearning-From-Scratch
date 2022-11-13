import tensorflow as tf
from keras.api._v2 import keras
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPool2D
from keras.models import Model


class QNet:
    def __init__(
            self, 
            hidden_size: int = 100, 
            output_size: int = 4) -> None:
        



