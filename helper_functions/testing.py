import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from helper_functions.nn_training import train_model
from keras.optimizers import Adam
from tensorflow.keras.models import clone_model
from helper_functions.nn import create_nn


def encode_and_split(X_tr, X_te):
    # Text portion of the training and test data
    X_train_text = X_tr[:, -1]
    X_test_text = X_te[:, -1]

    # Numerical portion of the training and test data
    X_train_num = X_tr[:, :-1].astype('float32')
    X_test_num = X_te[:, :-1].astype('float32')

    vectorizer = TfidfVectorizer(min_df=10)
    X_train_encoded = vectorizer.fit_transform(X_train_text).toarray().astype('float32')  # Fit the vectorizer and transform the training set

    X_test_encoded = vectorizer.transform(X_test_text).toarray().astype('float32')  # Transform the testing set
    
    return X_train_num, X_test_num, X_train_encoded, X_test_encoded

def cross_validate(X, y,
                   layer_values1 = [64, 32],
                   layers_value2 = [32],
                   dropout = 0,
                   num_folds=5, 
                   epochs=5,
                   opt = Adam(learning_rate=0.001, decay=1e-6),
                   batch_size = 32,
                   loss = 'binary_crossentropy'):
    
    '''
    Takes in a pre-compiled model, then returns the trained model for each fold of cross-validation
    with the given data type, along with the associated test data and labels, such that prediction
    can easily be performed for every model trained on each fold, and any given metric of interest
    can be computed
    '''
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Split data into folds
    fold_size = len(X) // num_folds
    fold_indices = [indices[i * fold_size: (i + 1) * fold_size] for i in range(num_folds)]

    models = []
    test_data = []
    test_labels = []

    # Perform cross-validation
    for i in range(num_folds):
        print(f'Fold {i+1}/{num_folds}')
        # Select validation set
        val_indices = fold_indices[i]
        X_val = X[val_indices]
        y_val = y[val_indices]

        # Select training set
        train_indices = np.concatenate([fold_indices[j] for j in range(num_folds) if j != i])
        X_train = X[train_indices]
        y_train = y[train_indices]

        
        X_train_num, X_val_num, X_train_encoded, X_val_encoded = encode_and_split(X_train, X_val)
        
        model = create_nn(layer_values1, layers_value2, X_train_encoded.shape[1], X_train_num.shape[1], dropout=dropout)

        model_copy = clone_model(model)
        
        model_copy = train_model(model_copy, 
                     [X_train_num, X_train_encoded], 
                     y_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     opt=opt,
                     loss=loss,
                     verbose=0,
                     )
        
        models.append(model_copy)
        
        test_data.append(([X_val_num, X_val_encoded]))
        test_labels.append(y_val)

    return models, test_data, test_labels