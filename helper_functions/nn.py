from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout

def create_nn(layer1_values, layer2_values, shape_text, shape_num, dropout=0):
    
    # Define input shapes
    text_input = Input(shape=(shape_text,), name='encoded_text_input')
    combined_input = Input(shape=(shape_num,), name='numeric_input')

    # Define neural network for text data with optional dropout
    for i in range(len(layer1_values)):
        if dropout > 0:
            text_model = Dense(layer1_values[i], activation='relu')(text_input)
            text_model = Dropout(dropout)(text_model)
        else:
            text_model = Dense(layer1_values[i], activation='relu')(text_input)

    # Concatenate text model output with combined numerical/categorical input
    combined_with_text = Concatenate()([combined_input, text_model])

    # Define additional layers if needed with optional dropout
    for i in range(len(layer2_values)):
        if dropout > 0:
            combined_with_text = Dense(layer2_values[i], activation='relu')(combined_with_text)
            combined_with_text = Dropout(dropout)(combined_with_text)
        else:
            combined_with_text = Dense(layer2_values[i], activation='relu')(combined_with_text)

    # Output layer
    output = Dense(1, activation='sigmoid')(combined_with_text)

    # Define model
    model = Model(inputs=[combined_input, text_input], outputs=output)
    
    return model
