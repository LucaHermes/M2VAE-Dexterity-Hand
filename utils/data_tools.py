import numpy as np
from scipy.special import cotdg

def periodic(x, phase_length=.25):
	sawtooth = -2/np.pi * np.arctan(cotdg(np.rad2deg((x*np.pi)/phase_length)))
	transform = (1 + sawtooth) / 2.
	return transform