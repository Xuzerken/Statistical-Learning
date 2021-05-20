import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import numpy as np
import os
import time
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

R_variable={}  ### used for saved all parameters and data
R_variable['input_dim']=1
R_variable['output_dim']=1
R_variable['train_size']=1000  ### training size
R_variable['batch_size']=R_variable['train_size'] # int(np.floor(R_variable['train_size'])) ### batch size
R_variable['test_size']=R_variable['train_size']  ### test size
R_variable['x_start']=-np.pi/2  #math.pi*3 ### start point of input
R_variable['x_end']=np.pi/2  #6.28/4 #math.pi*3  ### end point of input
R_variable['epoch_num']=300
R_variable['plotepoch']=50

def get_y_func(xs):
    return np.cos(xs)

R_variable['test_inputs']=np.random.rand(R_variable['test_size'],R_variable['input_dim'])*(R_variable['x_end']
-R_variable['x_start'])+R_variable['x_start']
R_variable['train_inputs']=np.random.rand(R_variable['train_size'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']

X_valid = R_variable['test_inputs']
X_train = R_variable['train_inputs']
y_valid = get_y_func(X_valid)
y_train = get_y_func(X_train)

class PlotCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch%(R_variable['plotepoch'])==0:
            self.ploty(epoch)
    
    def save_fig(self,fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
        plt.savefig(path, format=fig_extension, dpi=resolution)

    def ploty(self,pic_id):
        y_predict = self.model.predict(X_valid)
        if R_variable['input_dim']==1:
            plt.figure(pic_id)
            plt.grid()
            plt.scatter(X_valid.reshape(R_variable['test_size']),y_predict.reshape(R_variable['test_size']),c='r',s=10,label='Test')
            plt.scatter(X_train.reshape(R_variable['train_size']),y_train.reshape(R_variable['train_size']),c='g',s=10,label='Train')
            plt.scatter(X_valid.reshape(R_variable['test_size']),y_valid.reshape(R_variable['test_size']), c='b',s=10,label='True')
            self.save_fig("predict_plot_"+"%s"%(pic_id))
 
plot_cb = PlotCallback()

model = keras.models.Sequential([
    keras.layers.Dense(200, activation="tanh", input_shape=X_train.shape[1:]),
    keras.layers.Dense(200,activation="tanh"),
    keras.layers.Dense(200,activation="tanh"),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=1e-3))
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=R_variable['epoch_num'], validation_data=(X_valid, y_valid),callbacks=[checkpoint_cb,plot_cb], batch_size=R_variable['batch_size'])

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.6) # set the vertical range to [0-1]
plt.save_fig("loss")