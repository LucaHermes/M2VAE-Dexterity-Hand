from scipy.stats import pearsonr
import numpy as np
import pandas as pd

import vae_tools
import vae_tools.viz
import vae_tools.sanity
vae_tools.sanity.check()
import vae_tools.callbacks
from vae_tools.mmvae import MmVae, ReconstructionLoss
from keras.layers import Input, Dense, Lambda, Layer, Conv1D, Reshape, Flatten


#def build_mmvae(beta_norm, latent_dim, data_dimensions)

def pearson(A, B, n_samples=None):
	'''
	Correlates the features between two multidimensional datasets.
	'''
	a_dim = len(A[0])
	b_dim = len(B[0])
	
	corr = np.zeros([a_dim, b_dim])
	p_values = np.zeros([a_dim, b_dim])

	for i in range(a_dim):
		for j in range(b_dim):
			c, p = pearsonr(A[:n_samples, i], B[:n_samples, j])
			corr[i][j] = c
			p_values[i][j] = p

	return corr, p_values

#def evaluate(touch, proprioceptive, pose, latent_dims, beta_norms=[0.01], load_weights=True):
	


if __name__ == '__main__':
	z_dim = 56

	# read data
	touch_data = pd.read_csv('data/normalized_touch_data').to_numpy()
	proprio_data = pd.read_csv('data/normalized_proprio_data').to_numpy()
	quat_data = pd.read_csv('data/unnormalized_object_quat_data').to_numpy()
	#euler_angles = pd.read_csv('data/euler_angles').to_numpy()

	# build encoders
	dims = len(quat_data[0])
	batch_size = 128
	hidden_dims = 32
	beta_norm = 0.01
	epochs = 5
	load_weights = False
	# pose model
	quat_encoder = [[
	    Input(shape=(dims,)),
	    Dense(hidden_dims, activation='relu'),
	    Dense(hidden_dims//2, activation='relu'),
	]]

	quat_decoder = [[
	    Dense(hidden_dims//2, activation='relu'),
	    Dense(hidden_dims, activation='elu'),
	    Dense(dims, activation=None),
	]]

	touch_vae = MmVae(z_dim, quat_encoder, quat_decoder, [dims], 
		beta_norm, beta_is_normalized=True, name='quat_vae', 
		reconstruction_loss_metrics=[ReconstructionLoss.MSE])

	vae = touch_vae.get_model()
	vae.compile(optimizer='rmsprop')

	# losses_cb = vae_tools.callbacks.Losses(data=quat_data)

	if load_weights:
	    vae.load_weights('models/param_tuning/object_pose/object_pose')
	else:
	    vae.fit(quat_data, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1)
	    vae.save_weights('models/param_tuning/object_pose/object_pose')

	# build touch model
	dims = len(touch_data[0])
	batch_size = 512
	hidden_dims = 128
	beta_norm = 0.08
	epochs = 10
	load_weights = False

	touch_encoder = [[
    	Input(shape=(dims,)),
	    #Reshape((dims, 1)),
	    #Conv1D(128, 7, 4, activation='relu'),
	    #Flatten(),
	    Dense(hidden_dims, activation='relu'),
	    Dense(hidden_dims, activation='relu'),
	    Dense(hidden_dims//2, activation='relu'),
	    Dense(hidden_dims//4, activation='relu'),
	]]

	touch_decoder = [[
	    Dense(hidden_dims//2, activation='elu'),
	    Dense(hidden_dims, activation='elu'),
	    Dense(dims, activation='sigmoid'),
	]]

	touch_vea = MmVae(z_dim, touch_encoder, touch_decoder, [dims], beta_norm, beta_is_normalized=False,
	                  reconstruction_loss_metrics=[ReconstructionLoss.MSE], name='touch_vae')
	vae = touch_vea.get_model()
	vae.compile(optimizer='adam')

	losses_cb = vae_tools.callbacks.Losses(data=touch_data)

	if load_weights:
	    vae.load_weights('models/param_tuning/touch/touch')
	else:
	    vae.fit(touch_data, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1)
	    vae.save_weights('models/param_tuning/touch/touch')


	batch_size = 128
	dims = len(proprio_data[0])
	hidden_dims = 32
	beta_norm = 0.01
	epochs = 5
	load_weights = False

	proprio_encoder = [[
	    Input(shape=(dims,)),
	    Dense(hidden_dims, activation='elu'),
	    Dense(hidden_dims//2, activation='elu'),
	]]

	proprio_decoder = [[
	    Dense(hidden_dims//2, activation='elu'),
	    Dense(hidden_dims, activation='elu'),
	    Dense(dims, activation='tanh'),
	]]

	proprio_vea = MmVae(z_dim, proprio_encoder, proprio_decoder, [dims], beta_norm, beta_is_normalized=True,
	                  reconstruction_loss_metrics=[ReconstructionLoss.MSE], name='proprio_vae')
	vae = proprio_vea.get_model()
	vae.compile(optimizer='adam')

	losses_cb = vae_tools.callbacks.Losses(data=proprio_data)


	if load_weights:
	    vae.load_weights('models/param_tuning/proprio/proprio')
	else:
	    vae.fit(proprio_data, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1)
	    vae.save_weights('models/param_tuning/proprio/proprio')

	    vae_tools.viz.plot_losses(losses_cb, plot_elbo=True)

