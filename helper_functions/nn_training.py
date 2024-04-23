from keras.optimizers import Adam

def train_model(model, X_data, Y_data, 
             opt = Adam(learning_rate=0.001, decay=1e-6), 
             loss = 'binary_crossentropy', 
             callback=None, 
             batch_size=32,
             epochs=10,
             verbose=1):
    
    model.compile(optimizer=opt, 
                    loss=loss, 
                    metrics=['accuracy'])

    # Train the model
    if callback:
        model.fit(
            X_data,         # Input features
            Y_data,         # Target variable
            epochs=epochs,  # Number of epochs
            batch_size=batch_size,   # Batch size
            callbacks = [callback],
            verbose = verbose,
            )
    else:
        model.fit(
            X_data,         # Input features
            Y_data,         # Target variable
            epochs=epochs,  # Number of epochs
            batch_size=batch_size,   # Batch size
            verbose = verbose,
            )
        
    return model