def make_model(filter1=16, act1='relu', filter2=32, act2='relu', do1=.25, do2=.5, dense=32):
    input_shape = (size, size, 1)
    cnn = Sequential([
        Conv2D(filters=filter1,
               kernal_size=3,
               padding='same',
               activation=act1,
               input_shape=input_shape,
               name='CONV1'),
        Conv2D(filters=filter2,
               kernel_size=3,
               padding='same',
               activation=act2,
               name'CONV2'),
        MaxPooling2D(pool_size=2, name='POOL2'),
        Dropout(do1, name='DROP1),
        Flatten(name='FLAT1'),
        Dense(dense, activation='relu', name='FC1')
    ])
    cnn.compile(loss='mse',
                optimizer=tf.keras.optimzers.SGD(learning_rate=0.01,
                                                 momentum=0.9,
                                                 nesterov=False,
                                                 name='SGD'),
                metrics=[tf.keras.metrics.RootMeanSquareError(name='rmse')])
    return cnn                  
