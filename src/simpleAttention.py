from keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import initializers 

class Attention(Layer):
    def __init__(self, W_reg='l2', b_reg='l2', W_constraint='MinMaxNorm', b_constraint='MinMaxNorm', output_attention=False, **kwargs):
        self.initializer = initializers.GlorotUniform()
        self.weight_regularizers = regularizers.get(W_reg)
        self.bias_regularizers = regularizers.get(b_reg)
        self.weight_constraint = constraints.get(W_constraint)
        self.bias_constraint = constraints.get(b_constraint)
        self.output_attention = output_attention
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.weights = self.add_weight((input_shape[-1],), \
                initializer=self.init, name='{}_W'.format(self.name), \
                regularizer=self.W_regularizer, constraint=self.W_constraint)
        
        self.bias = self.add_weight((input_shape[1],), initializer='zero', \
            name='{}_b'.format(self.name), regularizer=self.b_regularizer, \
            constraint=self.b_constraint)
        
        self.built = True
    
    def compute_mask(self, input, input_mask=None):
        return None
    
    def call(self, x, mask=None):
        eij = dot_product(x, self.weights) + self.bias
        
        eij_active = K.tanh(eij)

        attention = K.exp(eij_active)

        if mask is not None:
            attention *= K.cast(mask, K.floatx())
        
        attention /= K.cast(K.sum(attention, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(attention)

        output = K.sum(weighted_input, axis=1)

        if self.output_attention:
            return [output, attention]
        return output

    def compute_output_shape(self, input_shape):
        if self.output_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


def dot_product(x, kernel):
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    
