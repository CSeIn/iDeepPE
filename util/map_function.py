## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.


from scipy.stats import skew
from tqdm import tqdm
import numpy as np
import scipy.special as spsp
import torch
import torch.nn as nn

def map_selector(map_type, params):
	"""
	"""
	if map_type == "Linear":
		return Linear(map_type)
	elif map_type == "DB":
		return DB(map_type)
	elif "Clip" in map_type:
		return Clip(map_type, params)
	elif "Logistic" in map_type:
		return Logistic(map_type, params)
	elif "Standardise" in map_type:
		return Standardise(map_type, params)
	elif "MinMaxScaling" in map_type:
		return MinMaxScaling(map_type, params)
	elif "NormalCDF" in map_type:
		return NormalCDF(map_type)
	elif "TruncatedLaplaceCDF" in map_type:
		return TruncatedLaplaceCDF(map_type, params)
	elif "LaplaceCDF" in map_type:
		return LaplaceCDF(map_type, params)
	elif "UniformCDF" in map_type:
		return UniformCDF(map_type, params)
	elif "Square" in map_type:
		return Square(map_type)
	# elif "TruncatedDoubleGammaCDF" in map_type:
	# 	return TruncatedDoubleGammaCDF(map_type, params)
	else: raise ValueError("Invalid map_type.")

class Map():
	"""
	Base map class.
	"""
	def __init__(self, map_type, params=None):
		"""
		Argument/s:

		"""
		self.map_type = map_type
		self.ten = torch.tensor(10.0).cuda()
		self.one = torch.tensor(1.0).cuda()

		if isinstance(params, list):
			self.params = [torch.tensor(param)  if param is not None else param for param in params]
		else:
			self.params = torch.tensor(params)  if params is not None else params

	def db(self, x):
		"""
		Converts power value to power in decibels.

		Argument/s:
			x - power value.

		Returns:
			power in decibels.
		"""
		x = torch.clamp(x, min=1e-12)
		return torch.mul(self.ten, torch.div(torch.log(x), torch.log(self.ten)))

	def db_inverse(self, x_db):
		"""
		Converts power in decibels to power value.

		Argument/s:
			x_db - power in decibels.

		Returns:
			x - power value.
		"""
		return torch.pow(self.ten, torch.div(x_db, self.ten))

	def stats(self, x):
		"""
		The base stats() function is used when no statistics are requied for
		the map function.

		Argument/s:
			x - a set of samples.
		"""
		pass

class Linear(Map):
	"""
	Linear map, i.e. no map.
	"""
	def map(self, x):
		"""
		Returns input.

		Argument/s:
			x - value.

		Returns:
			x.
		"""
		return x

	def inverse(self, x):
		"""
		Returns input.

		Argument/s:
			x - value.

		Returns:
			x.
		"""
		return x

class Square(Map):
	"""
	Square map.
	"""
	def map(self, x):
		"""
		Returns input.

		Argument/s:
			x - value.

		Returns:
			x^2.
		"""
		x_bar = torch.square(x)
		if 'DB' in self.map_type: x_bar = self.db(x_bar)
		return x_bar

	def inverse(self, x_bar):
		"""
		Returns input.

		Argument/s:
			x_bar - square value.

		Returns:
			x.
		"""
		if 'DB' in self.map_type: x_bar = self.db_inverse(x_bar)
		x = torch.sqrt(x_bar).numpy()
		return x

class Clip(Map):
	"""
	Clip values exceeding threshold. It depends on two parameters: min, max.
	Parameters are given using self.params=[min,max].
	"""
	def map(self, x):
		"""
		Returns clipped input.

		Argument/s:
			x - value.

		Returns:
			x_bar - clipped value.
		"""
		MIN, MAX = self.params
		x_bar = torch.clamp(x, min=MIN, max=MAX)
		if 'Square' in self.map_type: x_bar = torch.square(x_bar)
		if 'DB' in self.map_type: x_bar = self.db(x_bar)
		return x_bar

	def inverse(self, x):
		"""
		Returns input.

		Argument/s:
			x - value.

		Returns:
			x.
		"""
		if 'DB' in self.map_type: x = self.db_inverse(x)
		if 'Square' in self.map_type: x = torch.sqrt(x).numpy()
		return x

class DB(Map):
	"""
	Decibels map. Assumes input is power value.
	"""
	def map(self, x):
		"""
		Returns decibel value input.

		Argument/s:
			x - power value.

		Returns:
			x_bar - power value in decibels.
		"""
		return self.db(x)

	def inverse(self, x_bar):
		"""
		Inverse of decibel value.

		Argument/s:
			x_bar - power value in decibels.

		Returns:
			x - power value.
		"""
		return self.db_inverse(x_bar).numpy()

class Logistic(Map):
	"""
	Logistic map. It depends on two parameters: k, x_0. Parameters are given
	using self.params=[k,x_0].
	"""
	def map(self, x):
		"""
		Applies logistic function to input.

		Argument/s:
			x - value.

		Returns:
			f(x) - mapped value.
		"""
		k, x_0 = self.params
		if 'DB' in self.map_type: x = self.db(x)
		v_1 = torch.negative(torch.mul(k, torch.subtract(x, x_0)))
		return torch.reciprocal(torch.add(self.one, torch.exp(v_1)))

	def inverse(self, x_bar):
		"""
		Applies inverse of logistic map.

		Argument/s:
			x_bar - mapped value.

		Returns:
			x - value.
		"""
		k, x_0 = self.params
		v_1 = torch.subtract(torch.reciprocal(x_bar), self.one)
		v_2 = torch.log(torch.clamp(v_1, min=1e-12))
		x = torch.subtract(x_0, torch.mul(torch.reciprocal(k), v_2))
		if 'DB' in self.map_type: x = self.db_inverse(x)
		return x.numpy()

class Standardise(Map):
	"""
	Convert distribution to a standard normal distribution.
	"""
	def map(self, x):
		"""
		Normalise to a standard normal distribution.

		Argument/s:
			x - random variable realisations.

		Returns:
			x_bar.
		"""
		if 'Square' in self.map_type: x =  torch.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		x_bar = torch.div(torch.subtract(x, self.mu), self.sigma)
		return x_bar

	def inverse(self, x_bar):
		"""
		Inverse of normal (Gaussian) cumulative distribution function (CDF).

		Argument/s:
			x_bar - cumulative distribution function value.

		Returns:
			Inverse of CDF.
		"""
		x = torch.add(torch.mul(x_bar, self.sigma), self.mu)
		if 'DB' in self.map_type: x = self.db_inverse(x)
		if 'Square' in self.map_type: x = torch.sqrt(x)
		return x.numpy()

	def stats(self, x):
		"""
		Compute stats for each frequency bin.

		Argument/s:
			x - sample.
		"""
		if 'Square' in self.map_type: x =  torch.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		self.mu = torch.mean(x, dim=0)
		self.sigma = torch.std(x, dim=0)

class MinMaxScaling(Map):
	"""
	Normalise distribution between 0 and 1 using min-max scaling.
	"""
	def map(self, x):
		"""
		Normalise between 0 and 1.

		Argument/s:
			x - random variable realisations.

		Returns:
			x_bar.
		"""
		if 'Square' in self.map_type: x = torch.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		x_bar = torch.div(torch.subtract(x, self.min),
			torch.subtract(self.max, self.min))
		x_bar = torch.clamp(x_bar, min=0.0, max=1.0)
		return x_bar

	def inverse(self, x_bar):
		"""
		Inverse of max-min scaling.

		Argument/s:
			x_bar - max-min scaled value.

		Returns:
			Inverse of x_bar.
		"""
		x = torch.add(torch.mul(x_bar, torch.subtract(self.max,
			self.min)), self.min)
		if 'DB' in self.map_type: x = self.db_inverse(x)
		if 'Square' in self.map_type: x = torch.sqrt(x)
		return x.numpy()

	def stats(self, x):
		"""
		Compute stats for each frequency bin.

		Argument/s:
			x - sample.
		"""
		if 'Square' in self.map_type: x =  torch.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		self.min = torch.min(x, dim=0)
		self.max = torch.max(x, dim=0)

class NormalCDF(Map):
	"""
	Normal cumulative distribution function (CDF) map.
	"""
	def map(self, x):
		"""
		Normal (Gaussian) cumulative distribution function (CDF).

		Argument/s:
			x - random variable realisations.

		Returns:
			CDF.
		"""
		if 'Square' in self.map_type: x =  torch.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		v_1 = torch.subtract(x, self.mu)
		v_2 = torch.mul(self.sigma, torch.sqrt(torch.tensor(2.0)))
		v_3 = torch.erf(torch.div(v_1, v_2))
		return torch.mul(torch.tensor(0.5), torch.add(torch.tensor(1.0), v_3))

	def inverse(self, x_bar):
		"""
		Inverse of normal (Gaussian) cumulative distribution function (CDF).

		Argument/s:
			x_bar - cumulative distribution function value.

		Returns:
			Inverse of CDF.
		"""
		v_1 = torch.mul(self.sigma, torch.sqrt(torch.tensor(2.0)))
		v_2 = torch.mul(torch.tensor(2.0), x_bar)
		v_3 = torch.erfinv(torch.subtract(v_2, torch.tensor(1.0)))
		v_4 = torch.mul(v_1, v_3)
		x = torch.add(v_4, self.mu)
		if 'DB' in self.map_type: x = self.db_inverse(x)
		if 'Square' in self.map_type: x = torch.sqrt(x)
		return x

	def stats(self, x):
		"""
		Compute stats for each frequency bin.

		Argument/s:
			x - sample.
		"""
		if 'Square' in self.map_type: x =  torch.square(x)
		if 'DB' in self.map_type: x = self.db(x)
		self.mu = torch.mean(x, dim=0)
		self.sigma = torch.std(x, dim=0)

class LaplaceCDF(Map):
	"""
	Laplace cumulative distribution function (CDF) map. It depends
	on two parameters: mu and b. Parameters are given using
	self.params=[mu], and b is found from a sample of the training
	set.

	Parameter description:
		mu - location parameter.
		b - scale parameter.
	"""
	def map(self, x):
		"""
		Truncated Laplace cumulative distribution function (CDF).

		Argument/s:
			x - random variable realisations.

		Returns:
			x_bar - CDF.
		"""
		mu = self.params
		if 'DB' in self.map_type: x = self.db(x)
		x_bar = self.laplace_cdf(x, mu, self.b)
		return x_bar

	def inverse(self, x_bar):
		"""
		Inverse of truncated Laplace cumulative distribution function (CDF).

		Argument/s:
			x_bar - cumulative distribution function value.

		Returns:
			x - inverse of CDF value.
		"""
		mu = self.params
		x = self.laplace_cdf_inverse(x_bar, mu, self.b)
		if 'DB' in self.map_type: x = self.db_inverse(x)
		return x.numpy()

	def stats(self, x):
		"""
		Compute stats for each frequency bin.

		Argument/s:
			x - sample.
		"""
		mu = self.params
		if 'DB' in self.map_type: x = self.db(x)
		self.b = []
		for i in tqdm(range(x.shape[1])):
			x_k = x[:,i]
			mask = torch.greater(x_k, mu)
			x_k_right_tail = torch.subtract(torch.masked_select(x_k, mask), mu)
			self.b.append(torch.mean(x_k_right_tail, dim=0))
		self.b = np.array(self.b)

	def laplace_cdf(self, x, mu, b):
		"""
		Laplace cumulative distribution function (CDF).

		Argument/s:
			x - random variable realisations.
			mu - location parameter.
			b - scale parameter.

		Returns:
			CDF.
		"""
		v_1 = torch.subtract(x, mu)
		v_2 = torch.abs(v_1)
		v_3 = torch.negative(torch.div(v_2, b))
		v_4 = torch.exp(v_3)
		v_5 = torch.subtract(1.0, v_4)
		v_6 = torch.sign(v_1)
		v_7 = torch.mul(0.5, torch.mul(v_6, v_5))
		return torch.add(0.5, v_7)

	def laplace_cdf_inverse(self, cdf, mu, b):
		"""
		Inverse of Laplace cumulative distribution function (CDF).

		Argument/s:
			cdf - cumulative distribution function value.
			mu - location parameter.
			b - scale parameter.

		Returns:
			x - inverse of CDF.
		"""
		v_1 = torch.subtract(cdf, 0.5)
		v_2 = torch.abs(v_1)
		v_3 = torch.mul(2.0, v_2)
		v_4 = torch.subtract(1.0, v_3)
		v_5 = torch.log(v_4)
		v_6 = torch.sign(v_1)
		v_7 = torch.mul(b, torch.mul(v_6, v_5))
		return torch.subtract(mu, v_7)

class TruncatedLaplaceCDF(LaplaceCDF):
	"""
	Truncated Laplace cumulative distribution function (CDF) map. It depends
	on four parameters: mu, b, lower and upper. Parameters are given using
	self.params=[mu, lower, upper], and b is found from a sample of the training
	set.

	Parameter description:
		mu - location parameter.
		b - scale parameter.
		lower - lower limit.
		upper - upper limit.
	"""

	def map(self, x):
		"""
		Truncated Laplace cumulative distribution function (CDF).

		Argument/s:
			x - random variable realisations.

		Returns:
			x_bar - CDF.
		"""
		mu, lower, upper = self.params
		if 'DB' in self.map_type: x = self.db(x)
		x_bar_lower = self.laplace_cdf(lower, mu, self.b)
		x_bar_upper = self.laplace_cdf(upper, mu, self.b)
		x_bar = self.laplace_cdf(x, mu, self.b)
		x_bar = torch.div(torch.subtract(x_bar, x_bar_lower),
			torch.subtract(x_bar_upper, x_bar_lower))
		x_bar = torch.where(torch.less(x, lower), torch.zeros_like(x), x_bar)
		x_bar = torch.where(torch.greater(x, upper), torch.ones_like(x), x_bar)
		return x_bar

	def inverse(self, x_bar):
		"""
		Inverse of truncated Laplace cumulative distribution function (CDF).

		Argument/s:
			x_bar - cumulative distribution function value.

		Returns:
			x - inverse of CDF value.
		"""
		mu, lower, upper = self.params
		x_bar_lower = self.laplace_cdf(lower, mu, self.b)
		x_bar_upper = self.laplace_cdf(upper, mu, self.b)
		x_bar = torch.add(torch.mul(x_bar,
			torch.subtract(x_bar_upper, x_bar_lower)), x_bar_lower)
		x = self.laplace_cdf_inverse(x_bar, mu, self.b)
		if 'DB' in self.map_type: x = self.db_inverse(x)
		return x.numpy()

	def stats(self, x):
		"""
		Compute stats for each frequency bin.

		Argument/s:
			x - sample.
		"""
		mu, lower, upper = self.params
		if 'DB' in self.map_type: x = self.db(x)
		self.b = []
		for i in tqdm(range(x.shape[1])):
			x_k = x[:,i]
			mask = torch.logical_and(torch.greater(x_k, mu),
			 	torch.less(x_k, upper))
			x_k_right_tail = torch.subtract(torch.masked_select(x_k, mask), mu)
			self.b.append(torch.mean(x_k_right_tail, dim=0))
		self.b = np.array(self.b)

class UniformCDF(Map):
	"""
	Uniform cumulative distribution function (CDF) map. It depends
	on two parameters: a and b. Parameters are given using
	self.params=[a, b].

	Parameter description:
		a - lower limit.
		b - upper limit.
	"""
	def map(self, x):
		"""
		Applies uniform CDF to input.

		Argument/s:
			x - random variable realisations.

		Returns:
			x_bar - CDF.
		"""
		a, b = self.params
		return torch.div(torch.subtract(x, a),
			torch.subtract(b, a))

	def inverse(self, x_bar):
		"""
		Applies inverse of uniform CDF.

		Argument/s:
			x_bar - cumulative distribution function value.

		Returns:
			x - inverse of CDF value.
		"""
		a, b = self.params
		return torch.add(torch.mul(x_bar,
			torch.subtract(b, a)), a).numpy()