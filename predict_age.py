# Predict Biological Age for FSL vs Freesurfer Study
# Overleaf FSLvsFREESURFER_v3_ML

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, pdb
import datetime

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mutual_info_score
from sklearn.svm import SVR
from scipy.stats import spearmanr, pearsonr, linregress
from sklearn.decomposition import PCA
from sklearn import linear_model
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing



np.random.seed(11)
#params = {'activation':['relu'],'optimizer': ['Nadam', 'Adam'],'losses': ['mse'],'batch_size': [20,30,40],'epochs': [10,20]}
# Set figures dir
if sys.platform == 'linux':
	from numpy.random import default_rng
	rng = default_rng(42)  # Create a random number generator
	figures_dir = "/mnt/c/Users/borri/github/Quilis/FSLvsFREESURFER_v3_ML/Plots"
	csv_path = "/mnt/c/Users/borri/github/BBDD/Vallecas_Index-Vols1234567-Siena-Free-27ONovember2019_segmentation.csv"

else: 
	figures_dir = "C:\\Users\\borri\\github\\Quilis\\FSLvsFREESURFER_v3_ML\\Plots"
	csv_path = "C:\\Users\\borri\\github\\BBDD\\Vallecas_Index-Vols1234567-Siena-Free-27ONovember2019_segmentation.csv"

# Open csv dataset
dataframe = pd.read_csv(csv_path, sep=';') 
dataframe_orig = dataframe.copy()

fsl_cols = ['R_Thal_visita1','L_Puta_visita1','L_Amyg_visita1','R_Pall_visita1','L_Caud_visita1',\
'L_Hipp_visita1','R_Hipp_visita1','L_Accu_visita1','R_Puta_visita1','BrStem_visita1',\
'R_Caud_visita1','R_Amyg_visita1','L_Thal_visita1','L_Pall_visita1','R_Accu_visita1']

free_cols = ['fr_Right_Thalamus_Proper_y1','fr_Left_Putamen_y1','fr_Left_Amygdala_y1',\
'fr_Right_Pallidum_y1','fr_Left_Caudate_y1','fr_Left_Hippocampus_y1','fr_Right_Hippocampus_y1',\
'fr_Left_Accumbens_area_y1','fr_Right_Putamen_y1','fr_Right_Caudate_y1','fr_Right_Amygdala_y1',\
'fr_Left_Thalamus_Proper_y1','fr_Left_Pallidum_y1','fr_Right_Accumbens_area_y1']
demo_cols = ['sexo', 'apoe', 'dx_corto_visita1','edad_visita1']

dataframe = dataframe[fsl_cols + free_cols+ demo_cols]
dataframe_removenans = dataframe.dropna()

fsl_df =  dataframe_removenans[fsl_cols + demo_cols]
free_df = dataframe_removenans[free_cols + demo_cols]
# Functions HERE


def get_allsubmatrciesfrommatrix(df, dim):
	"""https://stackoverflow.com/questions/44111225/how-to-get-all-sub-matrices-of-2d-array-without-numpy
	"""
	#dim = 3
	indices = []
	submatrices = []
	matrix = df.to_numpy()
	for j in range(len(matrix)-(dim-1)):
		for i in range(len(matrix[0])-(dim-1)):
			# Get column name from indice
			#indices.append(df.columns[i])
			#indices.append(df.columns[j])
			X_i_j = [row[0+i:dim+i] for row in matrix[0+j:dim+j]]
			#print('== Indices =====')
			# el00,el01,el10,el11= [df.columns[i+1],df.columns[j]],[df.columns[i+1],df.columns[dim+j]] , [df.columns[dim+i-2],df.columns[j]], [df.columns[dim+i-2],df.columns[dim+j]]
			# print('Columns %s:' %df.columns[j],df.columns[dim+j])
			# print('Rows %s' %df.columns[i+1],df.columns[dim+i-2])
			# indices.append([el00,el01])
			# indices.append([el10,el11])
			submatrices.append(X_i_j)
			#for x in X_i_j:
			#	print(x)
	avg_matrices = np.average(submatrices[:])
	#print('Ag of matrices %.3f'%avg_matrices)
	return submatrices

def eigenvalues_analysis(cv, gamma, label=None):
	""""Calculate and show histogram of eigenvalues"
	"""
	print('Getting eigenvalues and eigenvectors...\n')
	ee, ev=np.linalg.eigh(cv)
	print('Eigenvalues of %s matrix are: %s' %(label, str(ee)))
	print('Çalling to EE to Calculate and show histogram of eigenvalues')
	EE(cv, gamma, label)
	return ee, ev



def marcpast_pde(l, g):
	"""http://www.bnikolic.co.uk/blog/python/2019/11/28/marchenko-pastur.html
		lambda and gamma is n/p= observations/features ~ 4000/14
	"""
	print("Marchenko-Pastur distribution")
	def m0(a):
		"Element wise maximum of (a,0)"
		return np.maximum(a, np.zeros_like(a))
	#gamma max
	gplus=(1+g**0.5)**2
	#gamma min
	gminus=(1-g**0.5)**2
	return np.sqrt( m0(gplus  - l) *  m0(l- gminus)) / ( 2*np.pi*g*l)


def EE(a, gamma, label=None):
	"Calculate and show histogram of eigenvalues"
	fig, ax = plt.subplots()
	ee, ev=np.linalg.eigh(a)
	nn, bb, patches=plt.hist(ee.ravel(), bins="auto", density=True)
	x=np.arange(bb[0], bb[-1], 0.003)

	plt.plot(x, marcpast_pde(x, gamma))
	plt.ylim(top=nn[1:].max() * 1.1)
	fig_name = os.path.join(figures_dir,  label + '_MarcenkoPastur.png')
	plt.savefig(fig_name)

def MarcenkoPasturPDF(var, qq, pts):
	"""
	"""

	eMin,eMax=var*(1-(1./qq)**.5)**2,var*(1+(1./qq)**.5)**2
	eVal=np.linspace(eMin,eMax,pts)
	pdf=qq/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
	#pdf=pd.Series(pdf,index=eVal)
	#rows = pdf.shape[0]
	pdf = pd.Series(pdf.reshape(pts,), index=eVal.reshape(pts,))
	return pdf

def fitKDE(obs,bWidth=.25,kernel='gaussian', x=None):
	# Fit kernel to a series of obs, and derive the prob of obs
	# x is the array of values on which the fit KDE will be evaluated
	from sklearn.neighbors.kde import KernelDensity
	if len(obs.shape)==1:obs=obs.reshape(-1,1)
	kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
	if x is None:x=np.unique(obs).reshape(-1,1)
	if len(x.shape)==1:x=x.reshape(-1,1)
	logProb=kde.score_samples(x) # log(density)
	pdf=pd.Series(np.exp(logProb),index=x.flatten())
	return pdf

def getPCA(matrix):
	# Get eVal,eVec from a Hermitian matrix
	eVal,eVec=np.linalg.eigh(matrix)
	indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
	eVal,eVec=eVal[indices],eVec[:,indices]
	eVal=np.diagflat(eVal)
	return eVal,eVec

def errPDFs(var,eVal,qq,bWidth,pts=1000):
	# Fit error
	pdf0=MarcenkoPasturPDF(var,qq,pts) # theoretical pdf
	pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values) # empirical pdf
	sse=np.sum((pdf1-pdf0)**2)
	return sse
#---------------------------------------------------
def findMaxEval(eVal,q,bWidth):
	# Find max random eVal by fitting Marcenko’s dist
	from scipy.optimize import minimize
	out=minimize(lambda *x:errPDFs(*x),.5,args=(eVal,q,bWidth),bounds=((1E-5,1-1E-5),))
	if out['success']:var=out['x'][0]
	else:var=1
	eMax=var*(1+(1./q)**.5)**2
	return eMax,var

def convertdf_intolongitudinal(df):
	""" Get df with measuremennt 1x N per years and transforms it longitudinal Nx1 
	"""
	# Create new dataframe for FSL and FreeSurfer
	frame_fsl1 = { 'age': df['edad_visita1'], 'sex': df['sexo'], 'fsl_R_Thal': df['R_Thal_visita1'], 'fsl_L_Thal': df['L_Thal_visita1'],'fsl_R_Puta': df['R_Puta_visita1'], 'fsl_L_Puta': df['L_Puta_visita1'],'fsl_R_Amyg': df['R_Amyg_visita1'], 'fsl_L_Amyg': df['L_Amyg_visita1'],'fsl_R_Pall': df['R_Pall_visita1'], 'fsl_L_Pall': df['L_Pall_visita1'],'fsl_R_Caud': df['R_Caud_visita1'], 'fsl_L_Caud': df['L_Caud_visita1'],'fsl_R_Hipp': df['R_Hipp_visita1'], 'fsl_L_Hipp': df['L_Hipp_visita1'],'fsl_R_Accu': df['R_Accu_visita1'], 'fsl_L_Accu': df['L_Accu_visita1']}
	frame_fsl2 = { 'age': df['edad_visita1'], 'sex': df['sexo'], 'fsl_R_Thal': df['R_Thal_visita2'], 'fsl_L_Thal': df['L_Thal_visita2'],'fsl_R_Puta': df['R_Puta_visita2'], 'fsl_L_Puta': df['L_Puta_visita2'],'fsl_R_Amyg': df['R_Amyg_visita2'], 'fsl_L_Amyg': df['L_Amyg_visita2'],'fsl_R_Pall': df['R_Pall_visita2'], 'fsl_L_Pall': df['L_Pall_visita2'],'fsl_R_Caud': df['R_Caud_visita2'], 'fsl_L_Caud': df['L_Caud_visita2'],'fsl_R_Hipp': df['R_Hipp_visita2'], 'fsl_L_Hipp': df['L_Hipp_visita2'],'fsl_R_Accu': df['R_Accu_visita2'], 'fsl_L_Accu': df['L_Accu_visita2']}
	frame_fsl3 = { 'age': df['edad_visita3'], 'sex': df['sexo'], 'fsl_R_Thal': df['R_Thal_visita3'], 'fsl_L_Thal': df['L_Thal_visita3'],'fsl_R_Puta': df['R_Puta_visita3'], 'fsl_L_Puta': df['L_Puta_visita3'],'fsl_R_Amyg': df['R_Amyg_visita3'], 'fsl_L_Amyg': df['L_Amyg_visita3'],'fsl_R_Pall': df['R_Pall_visita3'], 'fsl_L_Pall': df['L_Pall_visita3'],'fsl_R_Caud': df['R_Caud_visita3'], 'fsl_L_Caud': df['L_Caud_visita3'],'fsl_R_Hipp': df['R_Hipp_visita3'], 'fsl_L_Hipp': df['L_Hipp_visita3'],'fsl_R_Accu': df['R_Accu_visita3'], 'fsl_L_Accu': df['L_Accu_visita3']}
	frame_fsl4 = { 'age': df['edad_visita4'], 'sex': df['sexo'], 'fsl_R_Thal': df['R_Thal_visita4'], 'fsl_L_Thal': df['L_Thal_visita4'],'fsl_R_Puta': df['R_Puta_visita4'], 'fsl_L_Puta': df['L_Puta_visita4'],'fsl_R_Amyg': df['R_Amyg_visita4'], 'fsl_L_Amyg': df['L_Amyg_visita4'],'fsl_R_Pall': df['R_Pall_visita4'], 'fsl_L_Pall': df['L_Pall_visita4'],'fsl_R_Caud': df['R_Caud_visita4'], 'fsl_L_Caud': df['L_Caud_visita4'],'fsl_R_Hipp': df['R_Hipp_visita4'], 'fsl_L_Hipp': df['L_Hipp_visita4'],'fsl_R_Accu': df['R_Accu_visita4'], 'fsl_L_Accu': df['L_Accu_visita4']}
	frame_fsl5 = { 'age': df['edad_visita5'], 'sex': df['sexo'], 'fsl_R_Thal': df['R_Thal_visita5'], 'fsl_L_Thal': df['L_Thal_visita5'],'fsl_R_Puta': df['R_Puta_visita5'], 'fsl_L_Puta': df['L_Puta_visita5'],'fsl_R_Amyg': df['R_Amyg_visita5'], 'fsl_L_Amyg': df['L_Amyg_visita5'],'fsl_R_Pall': df['R_Pall_visita5'], 'fsl_L_Pall': df['L_Pall_visita5'],'fsl_R_Caud': df['R_Caud_visita5'], 'fsl_L_Caud': df['L_Caud_visita5'],'fsl_R_Hipp': df['R_Hipp_visita5'], 'fsl_L_Hipp': df['L_Hipp_visita4'],'fsl_R_Accu': df['R_Accu_visita5'], 'fsl_L_Accu': df['L_Accu_visita5']}
	frame_fsl6 = { 'age': df['edad_visita6'], 'sex': df['sexo'], 'fsl_R_Thal': df['R_Thal_visita6'], 'fsl_L_Thal': df['L_Thal_visita6'],'fsl_R_Puta': df['R_Puta_visita6'], 'fsl_L_Puta': df['L_Puta_visita6'],'fsl_R_Amyg': df['R_Amyg_visita6'], 'fsl_L_Amyg': df['L_Amyg_visita6'],'fsl_R_Pall': df['R_Pall_visita6'], 'fsl_L_Pall': df['L_Pall_visita6'],'fsl_R_Caud': df['R_Caud_visita6'], 'fsl_L_Caud': df['L_Caud_visita6'],'fsl_R_Hipp': df['R_Hipp_visita6'], 'fsl_L_Hipp': df['L_Hipp_visita6'],'fsl_R_Accu': df['R_Accu_visita6'], 'fsl_L_Accu': df['L_Accu_visita6']}
	fsl1, fsl2, fsl3, fsl4, fsl5, fsl6 = pd.DataFrame(frame_fsl1), pd.DataFrame(frame_fsl2), pd.DataFrame(frame_fsl3),  pd.DataFrame(frame_fsl4), pd.DataFrame(frame_fsl5), pd.DataFrame(frame_fsl6)
	df_fsl_lon = pd.concat([fsl1, fsl2, fsl3, fsl4, fsl5, fsl6])
	# Freesurfer
	frame_free1 = { 'age': df['edad_visita1'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y1'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y1'],'free_R_Puta': df['fr_Right_Putamen_y1'], 'free_L_Puta': df['fr_Left_Putamen_y1'],'free_R_Amyg': df['fr_Right_Amygdala_y1'], 'free_L_Amyg': df['fr_Left_Amygdala_y1'],'free_R_Pall': df['fr_Right_Pallidum_y1'], 'free_L_Pall': df['fr_Left_Pallidum_y1'],'free_R_Caud': df['fr_Right_Caudate_y1'], 'free_L_Caud': df['fr_Left_Caudate_y1'],'free_R_Hipp': df['fr_Right_Hippocampus_y1'], 'free_L_Hipp': df['fr_Left_Hippocampus_y1'],'free_R_Accu': df['fr_Right_Accumbens_area_y1'], 'free_L_Accu': df['fr_Left_Accumbens_area_y1']}
	frame_free2 = { 'age': df['edad_visita2'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y2'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y2'],'free_R_Puta': df['fr_Right_Putamen_y2'], 'free_L_Puta': df['fr_Left_Putamen_y2'],'free_R_Amyg': df['fr_Right_Amygdala_y2'], 'free_L_Amyg': df['fr_Left_Amygdala_y2'],'free_R_Pall': df['fr_Right_Pallidum_y2'], 'free_L_Pall': df['fr_Left_Pallidum_y2'],'free_R_Caud': df['fr_Right_Caudate_y2'], 'free_L_Caud': df['fr_Left_Caudate_y2'],'free_R_Hipp': df['fr_Right_Hippocampus_y2'], 'free_L_Hipp': df['fr_Left_Hippocampus_y2'],'free_R_Accu': df['fr_Right_Accumbens_area_y2'], 'free_L_Accu': df['fr_Left_Accumbens_area_y2']}
	frame_free3 = { 'age': df['edad_visita3'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y3'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y3'],'free_R_Puta': df['fr_Right_Putamen_y3'], 'free_L_Puta': df['fr_Left_Putamen_y3'],'free_R_Amyg': df['fr_Right_Amygdala_y3'], 'free_L_Amyg': df['fr_Left_Amygdala_y3'],'free_R_Pall': df['fr_Right_Pallidum_y3'], 'free_L_Pall': df['fr_Left_Pallidum_y3'],'free_R_Caud': df['fr_Right_Caudate_y3'], 'free_L_Caud': df['fr_Left_Caudate_y3'],'free_R_Hipp': df['fr_Right_Hippocampus_y3'], 'free_L_Hipp': df['fr_Left_Hippocampus_y3'],'free_R_Accu': df['fr_Right_Accumbens_area_y3'], 'free_L_Accu': df['fr_Left_Accumbens_area_y3']}
	frame_free4 = { 'age': df['edad_visita4'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y4'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y4'],'free_R_Puta': df['fr_Right_Putamen_y4'], 'free_L_Puta': df['fr_Left_Putamen_y4'],'free_R_Amyg': df['fr_Right_Amygdala_y4'], 'free_L_Amyg': df['fr_Left_Amygdala_y4'],'free_R_Pall': df['fr_Right_Pallidum_y4'], 'free_L_Pall': df['fr_Left_Pallidum_y4'],'free_R_Caud': df['fr_Right_Caudate_y4'], 'free_L_Caud': df['fr_Left_Caudate_y4'],'free_R_Hipp': df['fr_Right_Hippocampus_y4'], 'free_L_Hipp': df['fr_Left_Hippocampus_y4'],'free_R_Accu': df['fr_Right_Accumbens_area_y4'], 'free_L_Accu': df['fr_Left_Accumbens_area_y4']}
	frame_free5 = { 'age': df['edad_visita5'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y5'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y5'],'free_R_Puta': df['fr_Right_Putamen_y5'], 'free_L_Puta': df['fr_Left_Putamen_y5'],'free_R_Amyg': df['fr_Right_Amygdala_y5'], 'free_L_Amyg': df['fr_Left_Amygdala_y5'],'free_R_Pall': df['fr_Right_Pallidum_y5'], 'free_L_Pall': df['fr_Left_Pallidum_y5'],'free_R_Caud': df['fr_Right_Caudate_y5'], 'free_L_Caud': df['fr_Left_Caudate_y5'],'free_R_Hipp': df['fr_Right_Hippocampus_y5'], 'free_L_Hipp': df['fr_Left_Hippocampus_y5'],'free_R_Accu': df['fr_Right_Accumbens_area_y5'], 'free_L_Accu': df['fr_Left_Accumbens_area_y5']}
	frame_free6 = { 'age': df['edad_visita6'], 'sex': df['sexo'], 'free_R_Thal': df['fr_Right_Thalamus_Proper_y6'], 'free_L_Thal': df['fr_Left_Thalamus_Proper_y6'],'free_R_Puta': df['fr_Right_Putamen_y6'], 'free_L_Puta': df['fr_Left_Putamen_y6'],'free_R_Amyg': df['fr_Right_Amygdala_y6'], 'free_L_Amyg': df['fr_Left_Amygdala_y6'],'free_R_Pall': df['fr_Right_Pallidum_y6'], 'free_L_Pall': df['fr_Left_Pallidum_y6'],'free_R_Caud': df['fr_Right_Caudate_y6'], 'free_L_Caud': df['fr_Left_Caudate_y6'],'free_R_Hipp': df['fr_Right_Hippocampus_y6'], 'free_L_Hipp': df['fr_Left_Hippocampus_y6'],'free_R_Accu': df['fr_Right_Accumbens_area_y6'], 'free_L_Accu': df['fr_Left_Accumbens_area_y6']}
	free1, free2, free3, free4, free5, free6 = pd.DataFrame(frame_free1), pd.DataFrame(frame_free2), pd.DataFrame(frame_free3),  pd.DataFrame(frame_free4), pd.DataFrame(frame_free5), pd.DataFrame(frame_free6)
	df_free_lon = pd.concat([free1, free2, free3, free4, free5, free6])
	return df_fsl_lon, df_free_lon


def linear_reg_PCA(X,y):
	"""linear regression with x validation
		X is PCA output
	""" 
	
	from sklearn.model_selection import cross_val_predict
	from sklearn.metrics import mean_squared_error, r2_score

	regr = linear_model.LinearRegression()
	#pdb.set_trace()
	y = np.array(y).reshape(-1,1)
	
	regr.fit(X, y)

	y_c = regr.predict(X) 
	# x validation
	y_cv = cross_val_predict(regr, X, y, cv=5) 
	# Calculate scores for calibration and cross-validation 
	# Metrics for goodness of predction: R^2 and MSE
	score_c = r2_score(y, y_c) 
	score_cv = r2_score(y, y_cv)
	# Calculate mean square error for calibration and cross validation 
	mse_c = mean_squared_error(y, y_c) 
	mse_cv = mean_squared_error(y, y_cv)
	return regr


def plot_manifold(X_reduced, label=None):
	"""
	"""
	from matplotlib.ticker import NullFormatter
	fig, ax = plt.subplots()
	ax.scatter(X_reduced[:, 0], X_reduced[:, 1], cmap=plt.cm.Spectral)
	ax.set_title("Manifold Reduction: %s" %label)
	ax.xaxis.set_major_formatter(NullFormatter())
	ax.yaxis.set_major_formatter(NullFormatter())
	ax.axis('tight')
	fig_name = os.path.join(figures_dir, label + '_Manifold.png')
	plt.savefig(fig_name)

def manifold_analysis(X, label=None):
	"""LLE Locally Linear Embedding Dim Red technique
	"""
	
	from sklearn.manifold import LocallyLinearEmbedding
	from sklearn import manifold
	from time import time
	n_components = 2
	X = StandardScaler().fit_transform(X)
	print('Manifold Learning: LLE....\n')
	lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=10)
	t0 = time()
	X_reduced = lle.fit_transform(X)
	t1= time()
	print("%s: %.2g sec" % ('LLE', t1 - t0))
	plot_manifold(X_reduced, label +'_LLE')

	print('Manifold Learning: MDS....\n')
	mds = manifold.MDS(n_components, max_iter=100, n_init=1)
	t0 = time()
	X_reduced = mds.fit_transform(X)
	t1 = time()
	print("%s: %.2g sec" % ('MDS', t1 - t0))
	plot_manifold(X_reduced, label +'_MDS')


def PCA_analysis(df, label=None):
	""" df = dataframe_removenans[fsl_cols]
	"""

	n_components = 2
	# if n_components is not set all components are kept
	pca = PCA() 
	# Standardize features by removing the mean and scaling to unit variance
	Xstd = StandardScaler().fit_transform(df)
	# Run PCA producing the reduced variable Xreg and select the first pc components
	# Standarize input or not
	# We have to normalize the input data otherwise, FSL in 1 component get 90% of variance
	# https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca
	# Notmalization in important in PCA, because since PCA looks for maximizing the variance, if
	# one variable has very large variance will exploit it, without paying enough 
	#attention to the other variables, while if you normalize the results are more cogent
	Xreg = pca.fit_transform(Xstd)
	#Xreg = pca.fit(df)
	print('Explained Variance Vector is: \n')
	print(pca.explained_variance_ratio_)
	print('CumSum of Explained Variance Vector is:\n')
	print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))
	
	# Plot how many components are needed to describe the data. 
	fig, ax = plt.subplots()
	ax.plot(np.cumsum(pca.explained_variance_ratio_))
	ax.set_xlabel('number of components')
	ax.set_ylabel('cumulative explained variance')
	ax.set_title('PCA ' + label)
	ax.axis('tight')
	fig_name = label + '_PCA.png'
	fig_name = os.path.join(figures_dir, fig_name)
	plt.savefig(fig_name)
	return Xreg


def get_sklearn_train_test(X,y):
	"""
	"""
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	X_train_full, X_test, y_train_full, y_test  = train_test_split(X, y)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
	return X_train_full, X_test, y_train_full, y_test, X_train, X_valid, y_train, y_valid


def crossvalidation_keras(model, X,y):
	"""
	"""
	from sklearn.model_selection import KFold

	n_split = 5
	cv_mse = []
	history_list = []
	for train_index,test_index in KFold(n_split).split(X):
		X_train,X_test=X.iloc[train_index],X.iloc[test_index]
		y_train,y_test=y.iloc[train_index],y.iloc[test_index]
		history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, verbose=2)
		score = model.evaluate(X_test,y_test, verbose=0)
		cv_mse.append(score)
		history_list.append(history)
		# for different loss and metric
		#print(f'Test loss: {score[0]} / Test MAE: {score[1]}')
		print(f'Test loss: {score} ')

		# Visualize History
		print('Visualize History for X-Validation K==%i' %n_split)
		plot_loss(history)
	return history_list
	
def plot_loss(history):
	fig, ax = plt.subplots(figsize=(11,9))
	plt.plot(history.history['loss'], label='loss')
	plt.plot(history.history['val_loss'], label='val_loss')
	#plt.ylim([0, 10])
	plt.xlabel('Epoch')
	plt.ylabel('Error [Age]')
	plt.legend()
	plt.grid(True)
	

def build_and_compile_model(norm):
	from keras.regularizers import l1
	model = keras.Sequential([norm,layers.Dense(32, activation='sigmoid', kernel_regularizer=l1(0.01)),layers.Dense(64, activation='relu',kernel_regularizer=l1(0.01)),layers.Dense(1)])
	model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001),metrics=['mse', 'mae', 'mape']) #, metrics = ['mae']
	# Is possible to use loss != metric, but seems more reasonable use same loss and metric eg mae or mse
	#model.compile(loss='mse', optimizer= 'rmsprop', metrics = ['mae'])
	return model


def plot_DNN_predictions(y_test, y_pred, toolname=None):
	"""
	"""
	from sklearn.metrics import r2_score, explained_variance_score,max_error, mean_absolute_error
	
	fig, ax = plt.subplots(figsize=(11,9))
	today = datetime.date.today()
	original_stdout = sys.stdout
	a = plt.axes(aspect='equal')

	plt.scatter(y_test, y_pred)
	plt.xlabel('True Values [Age]')
	plt.ylabel('Predictions [Age]')
	plt.title(toolname + 'Predictions')
	lims = [68, 94]
	plt.xlim(lims)
	plt.ylim(lims)
	plt.text(90, 89, 'R2= %.3f' %round(r2_score(y_test, y_pred),3))
	_ = plt.plot(lims, lims)
	filename = os.path.join(figures_dir, toolname + ' predictions_scatter.png')
	plt.savefig(filename)

	# plot error predictions
	error = y_pred - y_test
	plt.hist(error, bins=25)
	plt.xlabel('Prediction Error [Age]')
	plt.title(toolname + 'Predictions')
	_ = plt.ylabel('Count')
	filename = os.path.join(figures_dir, toolname + 'predictions_error.png')
	plt.savefig(filename)

	# Call to regression metrics
	r2 = r2_score(y_test, y_pred)
	exvar = explained_variance_score(y_test, y_pred)
	maxerror = max_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	print('R2 score = %.3f Explained Variance = %.3f MaxError = %.3f MAE = %.3f' %(r2,exvar,maxerror,mae))
	# Write predictions file
	filename = os.path.join(figures_dir, toolname + '_DNN_report.txt')
	print('Writing DNN report to file: %s' %filename)
	print(today)
	print('MAE: %.3f' % mae)
	print('Max Error: %.3f' % maxerror)
	print('Expl Variance: %.3f' % exvar)
	print('R2: %.3f' % r2)
	with open(filename, 'w') as f:
		sys.stdout = f 
		print('Linear Regression Report for %s' %toolname)
		print(today)
		print('MAE: %.3f' % mae)
		print('Max Error: %.3f' % maxerror)
		print('Expl Variance: %.3f' % exvar)
		print('R2: %.3f' % r2)
		sys.stdout = original_stdout	

def feature_importance_eli(model, X, y, label=None):
	"""
	"""
	#from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
	import eli5
	from eli5.sklearn import PermutationImportance
	from IPython.display import display
	print('PermutationImportance with ELI5....\n')
	#Regression scores:https://scikit-learn.org/stable/modules/model_evaluation.html
	perm = PermutationImportance(model, scoring="explained_variance",random_state=1).fit(X,y)
	html_obj =eli5.show_weights(perm, feature_names = X.columns.tolist())
	result = pd.read_html(html_obj.data)[0]
	print('ELI5 Feature Importance \n')
	print(result)
	mask = perm.feature_importances_ > 0
	features = X.columns[mask]
	# Write html object to a file (adjust file path; Windows path is used here)
	htmlpath = os.path.join(figures_dir, label + '_eli5.htm')
	with open(htmlpath,'wb') as f:
		f.write(html_obj.data.encode("UTF-8"))
	return perm

def plot_metrics_trainval(history, metric, label=None):
	"""
	"""
	#plotter = tfdocs.plots.HistoryPlotter(metric ="mse", smoothing_std=10)
	fig, ax = plt.subplots(figsize=(11,9))
	plt.plot(history.history[metric]),plt.plot(history.history['val_' + metric])
	plt.legend(['train', 'val'], loc='upper right')
	plt.title(metric + ' '+ label)
	plt.ylabel(metric)
	plt.xlabel('epoch')
	plt.legend(['train', 'val'])
	fig_name = os.path.join(figures_dir, metric + ' ' + label +'_metrics_trainval.png')
	plt.savefig(fig_name)
	#plt.show(fig_name)

def compare_regressors(model, X, y, toolname=None):
	"""
	"""

	from sklearn.dummy import DummyRegressor
	from sklearn.metrics import mean_squared_error
	original_stdout = sys.stdout
	# Mean dummy predicitons
	dummy_regr = DummyRegressor(strategy="mean")
	dummy_regr.fit(X, y)
	dummy_regr.score(X, y)
	pred = dummy_regr.predict(X)
	dummy_regr.score(X, y)
	dummy_cte = np.sqrt(mean_squared_error(y,pred))
	# Random predictions
	pred_random = np.random.uniform(low=min(y),high=max(y),size=len(y))
	dummy_random = np.sqrt(mean_squared_error(y,pred_random))
	# Regressor predictions
	pred2 = model.predict(X)
	reg_real = np.sqrt(mean_squared_error(y,pred2))
	
	filename = filemodel = os.path.join(figures_dir, toolname+'_vs_dummy.txt')
	with open(filename, 'w') as f:
		sys.stdout = f 
		print('Dummy K MSE %.3f Dummy Random MSE %.3f Real Regressor MSE %.3f' %(dummy_cte, dummy_random, reg_real))
		sys.stdout = original_stdout

def DNN_regression_test(dataset, toolname=None):
	# reset index so we dont have repeated index for same subject in years
	#dataset = dataset.reset_index(drop=True)
	# Remove sex columns
	#dataset = dataset.drop('sex')
	import tensorflow_docs as tfdocs
	import tensorflow_docs.modeling
	import tensorflow_docs.plots
	from sklearn.metrics import mean_squared_error
	from math import sqrt
	train_dataset = dataset.sample(frac=0.8, random_state=0)
	test_dataset = dataset.drop(index=train_dataset.index)
	print(dataset.describe().transpose())
	print(train_dataset.describe().transpose())
	print(test_dataset.describe().transpose())
	
	# Split features from labels
	X_train = train_dataset.copy()
	X_test = test_dataset.copy()
	y_train = X_train.pop('age')
	y_test = X_test.pop('age')
	#Although a model might converge without feature normalization, normalization makes training much more stable.
	# Normalization Keras
	epochs = 1000
	normalizer = preprocessing.Normalization()
	normalizer.adapt(np.array(X_train))
	print(normalizer.mean.numpy())
	first = np.array(X_train[:1])
	with np.printoptions(precision=2, suppress=True):
		print('First example:', first)
		print()
		print('Normalized:', normalizer(first).numpy())

	# Talos hyper opt tool
	#talos_model(X_test, y_train, X_test,y_test, params)
	#scan_object = talos.Scan(dataset.drop(['age'], axis=1), dataset['age'], model=talos_model, params=params, experiment_name='talos_test', fraction_limit=0.1)
	size_histories = {}
	# Linear Model with Keras No Hidden Layers
	linear_model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])
	linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),loss='mean_absolute_error',metrics=['mse', 'mae', 'mape']) # metrics=['mae', 'mse']
	# Calculate validation results on 20% of the training data
	
	history = linear_model.fit(X_train, y_train, epochs=epochs,verbose=2, validation_split = 0.20)
	size_histories['shallow_NN'] = history
	plot_metrics_trainval(history, 'mae', toolname+'_shallow_NN')
	plot_metrics_trainval(history, 'mse', toolname+'_shallow_NN')
	plot_metrics_trainval(history, 'mape', toolname+'_shallow_NN')
	
	# Model weights
	print(linear_model.layers[1].kernel)
	# Predict 10 ages
	linear_model.predict(X_train[:10])
	test_results = {}
	test_results['linear_model'] = linear_model.evaluate(X_test, y_test, verbose=2)
	print('Comparing Deep Regressor with Dummy ones....\n')
	compare_regressors(linear_model,X_test,y_test,toolname+'_shallow_NN')

	# Compile Keras (Deep Network) Model
	dnn_model = build_and_compile_model(normalizer)
	dnn_model.summary()
	##### Cross validation  ##############################
	crossvalidation = False
	if crossvalidation is True:
		print('Running Cross Validation....\n')
		history_list = crossvalidation_keras(dnn_model, dataset.drop(['age'], axis=1), dataset['age']) 	
	#######################################################
	print('Training the DNN model.....\n')
	history = dnn_model.fit(X_train, y_train,validation_split=0.20,verbose=0, epochs=epochs)
	size_histories['deep_NN'] = history
	plot_metrics_trainval(history, 'mae', toolname+'_deep_NN')
	plot_metrics_trainval(history, 'mse', toolname+'_deep_NN')
	plot_metrics_trainval(history, 'mape', toolname+'_deep_NN')
	print('Comparing Deep Regressor with Dummy ones....\n')
	compare_regressors(linear_model,X_test,y_test,toolname+'_deep_NN')
	
	# Feature Importance with ELI5
	eliperm = feature_importance_eli(dnn_model, X_train,y_train, toolname +'deep_NN')

	pred_train, pred = dnn_model.predict(X_train), dnn_model.predict(X_test)
	print(' MRSE for Training data: %.3f' %np.sqrt(mean_squared_error(y_train,pred_train)))
	print(' MRSE for Test data: %.3f' %np.sqrt(mean_squared_error(y_test,pred)))
	
	plot_loss(history)

	test_results['dnn_model'] = dnn_model.evaluate(X_test, y_test, verbose=2)
	# Make predictions
	y_pred = dnn_model.predict(X_test).flatten()
	plot_DNN_predictions(y_test, y_pred, toolname)
	# Save the model
	filemodel = os.path.join(figures_dir, toolname+'dnn_model')
	dnn_model.save(filemodel)

	return size_histories
	# Re.load the model
	# reloaded = tf.keras.models.load_model(filemodel)
	# test_results['reloaded'] = reloaded.evaluate(X_test, y_test, verbose=2)


def MLP_regressor_test(X, y):
	"""https://www.tensorflow.org/tutorials/keras/regression
	"""

	print('Tensorflow version %s' %tf.__version__)
	
	# Get train valid and test datasets
	if 0==1:
		X_train_full, X_test, y_train_full, y_test, X_train, X_valid, y_train, y_valid = get_train_valid_test(X,y)
		# Standarization sklearn
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_valid_scaled = scaler.transform(X_valid)
		X_test_scaled = scaler.transform(X_test)
	else:
		print('Call to DNN_regression_test(dataset)')


def oneway_anovatest(data):
	"""https://ariepratama.github.io/How-to-Use-1-Way-Anova-in-Python/
	Test whether the mean of two groups (sex) are different
	"""
	import statsmodels.api as sm
	from statsmodels.formula.api import ols
	import scipy.stats as stats
	# Plot the volume distribution for the variable(volume str)
	fig, ax = plt.subplots(figsize=(11,9))
	plt.title('Subcortical Volume Distribution ')
	plt.ylabel('pdf')
	label = 'R_Thal_visita1' 
	labelC = 'sexo' #'sex'

	sns.distplot(data[label])
	# Plot by histogram by gender
	fig, ax = plt.subplots(figsize=(11,9))
	sns.distplot(data[data[labelC] == 1][label], ax=ax, label='Female')
	sns.distplot(data[data[labelC] == 0][label], ax=ax, label='Male')
	plt.title('Subcortical Volume for Each Gender')
	plt.legend()
	plt.show()

	data.groupby([labelC]).agg([np.mean, np.median, np.count_nonzero, np.std])[label]
	data = data.dropna()
	
	# Ordinary Least Squares (OLS) model

	model = ols('edad_visita1 ~ sexo', data=data).fit()
	anova_table = sm.stats.anova_lm(model, typ=2)
	anova_table


	fvalue, pvalue = stats.f_oneway(data[labelC], data[label])
	
	print(fvalue, pvalue)
	pdb.set_trace()

	# Null hypothesis vol male = vol female
	mod = ols('age ~ fsl_R_Thal', data=data[data['sex']==0]).fit()
	# do type 2 anova
	aov_table = sm.stats.anova_lm(mod, typ=2)
	print('ANOVA table for Male')
	print('----------------------')
	print(aov_table)

	mod = ols('age ~ fsl_R_Thal', data=data[data['sex']==1]).fit()
	# do type 2 anova
	aov_table = sm.stats.anova_lm(mod, typ=2)
	print('ANOVA table for Female')
	print('----------------------')
	print(aov_table)

	print()
	pdb.set_trace()


def test_linear_model_reg(X, y, toolname=None):
	"""
	"""
	from sklearn.metrics import max_error, explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
	today = datetime.date.today()
	
	original_stdout = sys.stdout
	filename = os.path.join(figures_dir, toolname + '_LinReg_report.txt')
	print('Calling to split_train_test for features %s and  label:%s' %(X.columns, y.name))
	
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	print('***Linear model to predict %s data' %y.name)
	# Fit the model real quick and compared with outliers
	model = linear_model.LinearRegression()
	model.fit(X_train, y_train)
	# evaluate the model
	yhat = model.predict(X_test)
	# evaluate predictions
	# Compute metrics
	mae = mean_absolute_error(y_test, yhat)
	md_sc = model.score(X_test, y_test)
	mse = mean_squared_error(y_test, yhat)
	eva = explained_variance_score(y_test, yhat)
	r2sc = r2_score(y_test, yhat)
	print('Writing LinReg report to file: %s' %filename)
	with open(filename, 'w') as f:
		sys.stdout = f 
		print('Linear Regression Report for %s' %toolname)
		print(today)
		print('MAE: %.3f' % mae)
		print('Model Score accuracy = %.4f' % md_sc)
		print('MSE: %.3f' % mae)
		print('Expl Variance: %.3f' % eva)
		print('R2: %.3f' % r2sc)
		
		# regression coefficients 	
		intercept = model.intercept_
		print("The intercept for our model is {}".format(intercept))
		for idx, col_name in enumerate(X_train.columns):
			print("The coefficient for {} is {}".format(col_name, model.coef_[idx]))
		sys.stdout = original_stdout

	
	return model	

def outlier_detection_isoforest(X, contamination, y=None):
	"""https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
	"""
	from sklearn.ensemble import IsolationForest
	#print('Outlier Detection Isoforest contamination rate =0.1')
	# contamination= [0.0, 0.5] 'auto'
	iso = IsolationForest(random_state=0, contamination=contamination)
	print(iso)
	yhat = iso.fit_predict(X)
	# select all rows that are not outliers
	mask = yhat != -1
	print('Number of outliers (rows) removed = %.d / %.d' %(sum(mask==False), yhat.shape[0]))
	if y is None:
		return  X[mask]
	else:
		X_train, y_train = X[mask], y[mask]
		return X_train, y_train

def test_compute_divergence(df_fsl_lon, df_free_lon):
	"""
	"""

	mis_dict = {'R_Thal':0, 'L_Thal':0,'R_Puta':0, 'L_Puta':0, 'R_Amyg':0, 'L_Amyg':0,'R_Pall':0, 'L_Pall':0, 'R_Caud':0, 'L_Caud':0, 'R_Accu':0, 'L_Accu':0, 'R_Hipp':0, 'L_Hipp':0}
	vols_dict = {'R_Thal':[], 'L_Thal':[],'R_Puta':[], 'L_Puta':[], 'R_Amyg':[], 'L_Amyg':[],'R_Pall':[], 'L_Pall':[], 'R_Caud':[], 'L_Caud':[], 'R_Accu':[], 'L_Accu':[], 'R_Hipp':[], 'L_Hipp':[]}
	for pair in mis_dict.keys():
		fsl_vol = 'fsl_' + pair
		free_vol = 'free_' + pair
		vol1, vol2 = df_fsl_lon[fsl_vol], df_free_lon[free_vol]
		volsdf = pd.concat([vol1,vol2],axis=1)
		volsdf = volsdf.dropna()
		print('Calling to oulier detection IsoForest #rows= %i' %volsdf.shape[0])
		volsdf = outlier_detection_isoforest(volsdf)
		print('DONE IsoForest Resulting #rows= %i' %volsdf.shape[0])
		print('Calling to MI for #rows = %i in Structure: %s' %(volsdf.shape[0], pair))
		mi = compute_divergence(volsdf.iloc[:,0], volsdf.iloc[:,1])
		print(' MI of %s == %.4f' %(pair, mi))
		mis_dict[pair]= mi
		# Add mean and std for F & F
		vols_dict[pair] = [vol1.mean(), vol1.std(),vol2.mean(),vol2.std()]

	return  mis_dict, vols_dict

def plot_pair_plots(df):
	"""https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166
	"""
	df_cl = outlier_detection_isoforest(df)

	sns.pairplot(df_cl, x_vars =  df_cl.iloc[:,2:].columns, y_vars = 'age', kind='reg')
	#g = sns.pairplot(df_cl, hue='sex')
	plt.show()
	

def compute_divergence(pse,qse):
	"""https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python
	KL divergence (and any other such measure) expects the input data to have a sum of 1
	"""
	
	return mutual_info_score(pse,qse)
	#return np.sum(np.where(pse !=0, pse*np.log(pse,qse),0))


def plot_bilateral_pairs(df_bilat_fsl, df_bilat_free, label):
	"""
	"""
	xlabels = df_bilat_fsl['str'].tolist()
	ylabels1 = df_bilat_fsl[label]
	ylabels2 = df_bilat_free[label]
	index = np.arange(len(xlabels))
	bar_width = 0.15
	fig, ax = plt.subplots()
	fsl = ax.bar(index, ylabels1, bar_width, label="fsl")
	free = ax.bar(index+ bar_width, ylabels2, bar_width,label="free")
	ax.set_xlabel('')
	ax.set_ylabel('Bilateral Correlation')
	ax.set_title(label + ' Bilateral Correlation FSLvsFreeS')
	ax.set_xticks(index + bar_width / 2)
	ax.set_xticklabels(xlabels)
	ax.legend()
	fig_name = label + 'bilat_corr.png'
	fig_name = os.path.join(figures_dir, fig_name)
	plt.savefig(fig_name)

def plot_bilat_results(df_bilat_fsl, df_bilat_free):
	"""
	"""
	fig, ax = plt.subplots(2)
	df_bilat_fsl.plot(ax =ax[0], x="str", y=["spearman", "kendall", "pearson"], use_index=False, fontsize=6, xticks=[], kind="bar",title='FSL')
	df_bilat_free.plot(ax =ax[1], x="str", y=["spearman", "kendall", "pearson"], fontsize=8, kind="bar",title='FreeSurfer',legend=False)
	fig_name = os.path.join(figures_dir, 'metricsfVSf.png')
	plt.savefig(fig_name)
	#plt.show()
	# Plot Pairs
	print('Calling to Plot bilateral Structures per method...\n')
	plot_bilateral_pairs(df_bilat_fsl, df_bilat_free, 'pearson')
	plot_bilateral_pairs(df_bilat_fsl, df_bilat_free,'kendall')
	plot_bilateral_pairs(df_bilat_fsl, df_bilat_free,'spearman')



def symmetry_study(df, label=None):
	"""
	"""
	import warnings 
	warnings.filterwarnings('ignore')
	sns.set(style="white")
	sns.set(color_codes=True)
	
	df = df.reset_index(drop=True)
	X = df.drop(['age'], axis=1)
	#y = df['age']

	# ax = sns.pairplot(df, hue="sex", diag_kind = 'kde', palette="husl") #, markers=["*", "+"]
	# fig_name = os.path.join(figures_dir, label + 'pairplot.png')
	# plt.savefig(fig_name)
	# print('Figure saved: %s' %fig_name)
	# #plt.show()
	
	# Compute correlation matrix
	cols = ['Thal', 'Puta','Amyg','Pall', 'Caud', 'Hipp', 'Accu']
	bilat_cols = {'str': pd.Series(cols), 'pearson': pd.Series(), 'kendall':pd.Series(), 'spearman':pd.Series() }
	df_bilat = pd.DataFrame(data=bilat_cols)
	df_bilat['pearson'].astype('float64').dtypes
	for col in cols:

		colR = label + '_R_'  + col
		colL = label + '_L_'  + col
		pearsond = df[[colR,colL]].corr(method='pearson').iloc[0,1]
		kendalld = df[[colR,colL]].corr(method='kendall').iloc[0,1]
		spearmand = df[[colR,colL]].corr(method='spearman').iloc[0,1]
		ix = df_bilat.index[df_bilat['str']==col][0]
		df_bilat.at[ix,'pearson'] = pearsond
		df_bilat.at[ix,'spearman'] = spearmand
		df_bilat.at[ix,'kendall'] = kendalld
	return df_bilat

def eda_plots(df, label=None):
	"""Plot EDA pandas df
	"""
	
	# Remove sex and age get 14 cols with subcortical volumes
	df = df.iloc[:,2:]
	#subcortical = df.columns[2:].to_list()
	subcortical = df.columns.to_list()
	print('Median volumes of %s \n'% label)
	sorted_nb = df[subcortical].median().sort_values()
	print(sorted_nb)
	f = plt.figure(figsize=(19, 15))
	# Plot volumes increasing order
	ax = df.boxplot(column=sorted_nb.index.values.tolist(), rot=45, fontsize=14)
	ax.set_title('Subcortical volume estimates ')
	#ax.set_xlabel(' ')
	ax.set_ylabel(r'Volume in $mm^3$')
	# Save fig
	fig_name = os.path.join(figures_dir, label + '_boxplot.png')
	plt.savefig(fig_name)
	# correlations
	f = plt.figure(figsize=(19, 15))
	plt.matshow(df.corr(method='pearson'), fignum=f.number)
	plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
	plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=14)
	#plt.title('Correlation Matrix', fontsize=16);
	fig_name = os.path.join(figures_dir, label + '_corr.png')
	plt.savefig(fig_name)

def plot_removed_foriso(indexes, fsl_counts, free_counts):
	""" Plot stacked bar wirh #removed/remain based on isoforest contamination param
	"""
	df = pd.DataFrame({'contamination': indexes, 'fsl removed':fsl_counts, 'free removed':free_counts}, index=indexes)
	ax = df.plot.bar(rot=0)
	ax.set_xlabel('IsoForest Contamination')
	ax.set_ylabel('$\\%$ removed cases')
	fig_name = os.path.join(figures_dir, 'PCremovedIsoforest.png')
	plt.savefig(fig_name)
	return df

def plot_corr_matrix(df, label=None, rows_col=None):
	"""
	"""
	plt.rcParams["axes.grid"] = False
	fig, ax = plt.subplots()
	fig.tight_layout() 
	data = df.corr(method='pearson')
	im = ax.matshow(data)
	plt.xticks(range(df.shape[1]), df.columns, fontsize=11, rotation=45)
	if rows_col is not None:
		# plot bilateral corr matrix
		plt.yticks(range(df.shape[1]), rows_col, fontsize=11)
	else:
		plt.yticks(range(df.shape[1]), df.columns, fontsize=11)
	#cb = ax.colorbar()
	#cb.ax.tick_params(labelsize=14)
	#plt.title('Correlation Matrix', fontsize=16);
	for (i, j), z in np.ndenumerate(data):
		ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
	fig_name = os.path.join(figures_dir, label + '_corr.png')
	fig.colorbar(im, orientation='vertical')
	plt.savefig(fig_name)

def plot_correlation_matrix_by_side(df, Ls, Rs, label=None):
	"""
	"""
	plt.rcParams["axes.grid"] = False
	fig, ax = plt.subplots()
	fig.tight_layout() 
	data= df.loc[Ls][Rs]
	im = ax.matshow(data)
	plt.xticks(range(len(Ls)), Ls, fontsize=11, rotation=45)
	plt.yticks(range(len(Rs)), Rs, fontsize=11)
	#for (i, j), z in np.ndenumerate(data):
	#	ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
	fig_name = os.path.join(figures_dir, label + '_corr_all.png')
	fig.colorbar(im, orientation='vertical')
	plt.savefig(fig_name)

def correlation_by_tool(dataset, cols, label=None):
	"""correlation_by_tool: all 2x2 matrices bith sides and bilateral
	"""
	R_cols = [s for s in cols if "_R_" in s]
	L_cols = [s for s in cols if "_L_" in s]
	matrix = dataset.drop(['age','sex'], axis=1)
	corrmatrix = matrix.corr(method='pearson')
	#####
	plot_correlation_matrix_by_side(corrmatrix, cols, cols, str(label)+'_T')
	plot_correlation_matrix_by_side(corrmatrix, L_cols, R_cols, str(label)+'_B')
	plot_correlation_matrix_by_side(corrmatrix, R_cols, R_cols, str(label)+'_R')
	plot_correlation_matrix_by_side(corrmatrix, L_cols, L_cols, str(label)+'_L')

	#matrix = dataset[cols].corr(method='pearson')
	R_matrix = corrmatrix.loc[R_cols][R_cols]
	L_matrix = corrmatrix.loc[L_cols][L_cols]
	bilat = corrmatrix.loc[L_cols][R_cols]
	# get all dimXdim matrices
	T = get_allsubmatrciesfrommatrix(corrmatrix,2)
	R = get_allsubmatrciesfrommatrix(R_matrix,2)
	L = get_allsubmatrciesfrommatrix(L_matrix,2)
	B = get_allsubmatrciesfrommatrix(bilat,2)
	T, R, L, B = np.average(T),np.average(R),np.average(L),np.average(B)
	#T, R, L,B = np.asarray(T), np.asarray(R),np.asarray(L),np.asarray(B)
	print('Tool:%s 2x2 submatrices T = %.4f R = %.4f L = %.4f  B = %.4f'%(label, T,R,L,B))
	return [label,T,R,L,B]

def correlation_by_sex(df, cols, label=None):
	"""Output: Females, Males
	"""
	cols.append('sex')

	dataF =  df[cols][df[cols]['sex']==1].drop('sex',axis=1)
	dataM =  df[cols][df[cols]['sex']==0].drop('sex',axis=1)
	females = dataF.corr(method='pearson')
	males = dataM.corr(method='pearson')
	#  Plot correlation matrices
	plot_corr_matrix(dataM, str(label)+'_males')
	plot_corr_matrix(dataF, str(label)+'_females')
	def correlation_by_sex_aux(df, cols, sex):
		#thisdict = {"brand": "Ford","model": "Mustang","year": 1964}
		R_cols = [s for s in cols if "_R_" in s]
		L_cols = [s for s in cols if "_L_" in s]
		#pdb.set_trace()
		T = df
		R = df.loc[R_cols][R_cols]
		L = df.loc[L_cols][L_cols]
		B = df.loc[L_cols][R_cols]
		plot_corr_matrix(R, sex+'_R')
		plot_corr_matrix(L, sex+'_L')
		plot_corr_matrix(B, sex+'_B', L_cols)
		#fsl_R_matrix = fsl_matrix[[fsl_R_cols][fsl_R_cols]]
		T = get_allsubmatrciesfrommatrix(T,2)
		R = get_allsubmatrciesfrommatrix(R,2)
		L = get_allsubmatrciesfrommatrix(L,2)
		B = get_allsubmatrciesfrommatrix(B,2)
		T,R,L,B= np.average(T[:]), np.average(R[:]), np.average(L[:]),np.average(B[:])
		print('Averages 2x2 for: %s is T =%.3f R =%.3f L =%.3f B =%.3f' %(sex,T,R,L,B))
		return [sex, T,R,L,B]
	male_r = correlation_by_sex_aux(males, cols, label+'_male')
	female_r = correlation_by_sex_aux(females, cols, label+'_female')
	return [male_r,female_r]


	
def main_test():
	"""
	"""

	fsl_lon_cols = ['fsl_R_Thal', 'fsl_L_Thal', 'fsl_R_Puta', 'fsl_L_Puta','fsl_R_Amyg', 'fsl_L_Amyg', 'fsl_R_Pall', 'fsl_L_Pall', 'fsl_R_Caud','fsl_L_Caud', 'fsl_R_Hipp', 'fsl_L_Hipp', 'fsl_R_Accu', 'fsl_L_Accu']
	free_lon_cols = ['free_R_Thal', 'free_L_Thal', 'free_R_Puta','free_L_Puta', 'free_R_Amyg', 'free_L_Amyg', 'free_R_Pall','free_L_Pall', 'free_R_Caud', 'free_L_Caud', 'free_R_Hipp','free_L_Hipp', 'free_R_Accu', 'free_L_Accu']
	fsl_R_cols = [s for s in fsl_lon_cols if "_R_" in s]
	fsl_L_cols = [s for s in fsl_lon_cols if "_L_" in s]
	free_R_cols = [s for s in free_lon_cols if "_R_" in s]
	free_L_cols = [s for s in free_lon_cols if "_L_" in s]
	# Get longitudinal dataframe
	#oneway_anovatest(dataframe_orig)
	df_fsl_lon, df_free_lon = convertdf_intolongitudinal(dataframe_orig)
	df_fsl_lon, df_free_lon = df_fsl_lon.dropna(), df_free_lon.dropna()
	fsl_totalr = df_fsl_lon.shape[0]
	free_totalr = df_free_lon.shape[0]
	contamination = [0.01, 0.05, 0.1, 'auto']
	fsl_datasets, free_datasets, indexes, fsl_counts, free_counts = [], [], [], [],[]
	for cont in contamination:
		print('outlier_detection_isoforest contamination =%s \n' %str(cont))
		df_fsl_lon = outlier_detection_isoforest(df_fsl_lon, cont)
		df_free_lon = outlier_detection_isoforest(df_free_lon, cont)
		# percetange of ramianing cases
		fsl_removed_pc = 1 - (df_fsl_lon.shape[0] - fsl_totalr)/100
		free_removed_pc =1 - (df_free_lon.shape[0] - free_totalr)/100
		fsl_datasets.append(df_fsl_lon), free_datasets.append(df_free_lon)
		indexes.append(cont), fsl_counts.append(fsl_removed_pc), free_counts.append(free_removed_pc)

	print('Plotting Bar plot removed_foriso...\n')
	df_plot = plot_removed_foriso(indexes, fsl_counts, free_counts)
	# Select auto contamination
	df_fsl_lon, df_free_lon = fsl_datasets[2], free_datasets[2]

	####################################################################
	## EDA 
	dataset = df_fsl_lon.reset_index(drop=True)
	dataset2 = df_free_lon.reset_index(drop=True)
	print('EDA plots for FSL...\n')
	eda_plots(dataset, 'iso_01_fsl')
	print('EDA plots for Freesurfer...\n')
	eda_plots(dataset2, 'iso_01_free')

	###
	####################################################################
	## Symmetry Study. Distance between hemispheres
	print('\n Symmetry Study for FSL.Computing distances between hemispheres... \n\n')
	df_bilat_fsl = symmetry_study(dataset, 'fsl')
	print('\n Symmetry Study *FSL* DONE ! \n')
	print('\n Symmetry Study for FREES.Computing distances between hemispheres... \n\n')
	df_bilat_free = symmetry_study(dataset2, 'free')
	print('\n Symmetry Study *FREESFR* DONE ! \n')
	plot_bilat_results(df_bilat_fsl, df_bilat_free)
	##
	####################################################################

	###
	####################################################################
	
	# Correlation matrix by tool and distinguish btw sex
	fsl_ = correlation_by_tool(dataset, fsl_lon_cols, 'fsl')
	free_ = correlation_by_tool(dataset2, free_lon_cols, 'free')
	pdb.set_trace()
	fsl_sex = correlation_by_sex(dataset, fsl_lon_cols, 'fsl')
	free_sex = correlation_by_sex(dataset2, free_lon_cols, 'free')
	pdb.set_trace()
	#Marcenko-Pastur Theorem
	gamma = dataset.shape[0]/ dataset.shape[1]
	eigenvalues_analysis(fsl_matrix,gamma,'fsl')
	eigenvalues_analysis(free_matrix,dataset2.shape[0]/dataset2.shape[1],'free')
	pdb.set_trace()
	#fsl_matrix = np.corrcoef(free_matrix,rowvar=0)
	ee, ev=np.linalg.eigh(fsl_matrix)  
	ee2, ev2 =np.linalg.eigh(free_matrix)  
	pdb.set_trace()
	
	EE(fsl_matrix, gamma)

	eVal_fsl, eVec_fsl = getPCA(fsl_matrix)
	eVal_free, eVec_free = getPCA(free_matrix)
	eMax0,var0 = findMaxEval(np.diag(eVal_fsl),dataset.shape[0]/dataset.shape[1],bWidth=.01)
	nFacts0 = eVal_fsl.shape[0]-np.diag(eVal_fsl)[::-1].searchsorted(eMax0)
	
	pdb.set_trace()
	eMax0_,var0_ = findMaxEval(np.diag(eVal_free),dataset2.shape[0]/dataset2.shape[1],bWidth=.01)
	nFacts0_ = eVal_free.shape[0]-np.diag(eVal_free)[::-1].searchsorted(eMax0_)

	###
	####################################################################

	####################################################################
	## Dimensionality Reduction 
	#dataset = df_fsl_lon.reset_index(drop=True)
	print('Calling to PCA for FSL cols\n')
	Xreg_fsl = PCA_analysis(dataset.drop(['age', 'sex'], axis=1), 'fsl')
	print('Calling to Manifold analysis for FSL cols\n')
	manifold_analysis(dataset.drop(['age', 'sex'], axis=1), 'fsl')
	# FreeSurfer
	#dataset2 = df_free_lon.reset_index(drop=True)
	print('Calling to PCA for FreeSurfer cols\n')
	Xreg_free = PCA_analysis(dataset2.drop(['age', 'sex'], axis=1), 'free')
	print('Calling to Manifold analysis for FreeSurfer cols\n')
	manifold_analysis(dataset2.drop(['age', 'sex'], axis=1), 'free')
	pdb.set_trace()
	###
	####################################################################

	##### TEST delete ####
	dataset = df_fsl_lon.reset_index(drop=True)
	dataset2 = df_free_lon.reset_index(drop=True)
	model1 = test_linear_model_reg(dataset.drop(['age', 'sex'], axis=1), dataset['age'], 'fsl')
	model2 = test_linear_model_reg(dataset2.drop(['age', 'sex'], axis=1), dataset2['age'], 'free')
	print('DNN for FSL....\n\n')
	DNN_regression_test(dataset, 'fsl')
	print('DNN for FREESURFER....\n\n')
	DNN_regression_test(dataset2, 'free')
	pdb.set_trace()
	
	####################################################################
	## Prediction LinReg and NN
	print('Calling to Linear Regression for FSL...\n')
	test_linear_model_reg(dataset, 'FSL')
	test_linear_model_reg(dataset2, 'FreeSurfer')
	
	print('DNN for FSL....\n\n')
	DNN_regression_test(df_fsl_lon)
	print('DNN for FREESURFER....\n\n')
	df_free_lon = outlier_detection_isoforest(df_free_lon)
	DNN_regression_test(df_free_lon)
	pdb.set_trace()
	# DNN regression test
	# compare metrics LR vs DNN

	###
	####################################################################

	#df_fsl_lon = outlier_detection_isoforest(df_fsl_lon.drop('sex', axis=1))

	## Test MLPRegressor Deep Learning
	X, y = df_fsl_lon[fsl_lon_cols], df_fsl_lon['age'] 
	MLP_regressor_test(X, y)


	# Remove NANS
	#print('Removing NaNs, prior #rows %i %i' %(df_fsl_lon.shape[0],df_free_lon.shape[0]) )
	#df_fsl_lon, df_free_lon = df_fsl_lon.dropna(), df_free_lon.dropna()
	#print('Removed NaNs, post #rows %i %i' %(df_fsl_lon.shape[0],df_free_lon.shape[0]) )
	print('Calling to MI between estimates, mis_dict and the mean and std of each structure')
	#mis_dict, vols_dict = test_compute_divergence(df_fsl_lon, df_free_lon)
	
	# YS: Plot MI and study corr between mean vol and MI

	# Get X and y

	X = dataframe_removenans[fsl_cols]
	y = dataframe_removenans['edad_visita1']
	X['age'] = y

	# Plot pairs as point cloud with regression line 
	df_fsl_lon = df_fsl_lon.dropna()
	#plot_pair_plots(df_fsl_lon)
	df_free_lon = df_free_lon.dropna()
	#plot_pair_plots(df_free_lon)
	

	# Split dataset
	#X_train_fsl, X_test_fsl, y_train_fsl, y_test_fsl = train_test_split(dataframe_removenans[fsl_cols], y, test_size=0.25, random_state=1)
	#X_train_free, X_test_free, y_train_free, y_test_free = train_test_split(dataframe_removenans[free_cols], y, test_size=0.25, random_state=1)
	

	X_train_fsl, X_test_fsl, y_train_fsl, y_test_fsl = train_test_split(df_fsl_lon[fsl_lon_cols], df_fsl_lon['age'], test_size=0.25, random_state=1)
	X_train_free, X_test_free, y_train_free, y_test_free = train_test_split(df_free_lon[free_lon_cols], df_free_lon['age'], test_size=0.25, random_state=1)
	# Detect outliers

	# Algorithms for outlier detection
	X_cl_fsl, y_cl_fsl = outlier_detection_isoforest(X_train_fsl, y_train_fsl)
	X_cl_free, y_cl_free = outlier_detection_isoforest(X_train_free, y_train_free)
	#X_cl_fsl, y_cl_fsl = outlier_detection_isoforest(dataframe_removenans[fsl_cols],y)
	#X_cl_free, y_cl_free = outlier_detection_isoforest(dataframe_removenans[free_cols],y)
	
	# Prediction
	# Linear Prediction
	test_linear_model_reg(X_cl_fsl, y_cl_fsl, X_test_fsl,y_test_fsl, 'FSL')
	test_linear_model_reg(X_cl_free, y_cl_free, X_test_free,y_test_free, 'FreeSurfer')
	pdb.set_trace()
	## Linear Regression
	## PCR
	test_linear_model_pcr(X_cl_free, y_cl_free, X_test_free,y_test_free, 'FreeSurfer')
	# Non Linear Neural Network
	test_linear_model()



	# Plot outliers
	sns.boxplot(x=X_train_fsl.iloc[:,0])
	plt.show()
	sns.boxplot(x=X_cl_fsl.iloc[:,0])
	plt.show()

	pdb.set_trace()









	#SPearman and Pearson correlation
	coef_s, pval_s = spearmanr(X.iloc[:,0], y)
	coef, pval = pearsonr(X.iloc[:,0], y)
	print('Pearson correlation coefficient and pvalue: %.5f %.5f' %(coef,pval))
	print('Spearmans correlation coefficient and pvalue: %.5f %.5f' %(coef_s,pval_s))
	# Using pandas

	pear = X.corr() 
	spear = X.corr(method='spearman')  # Spearman's rho
	fig, ax = plt.subplots()
	im = ax.imshow(pear)
	im.set_clim(-1, 1)
	ax.xaxis.set(ticklabels= X.columns)
	ax.yaxis.set(ticklabels=X.columns) #ticks=(0, 1, 2), 
	cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
	plt.show()

	# interpret the significance
	alpha = 0.05
	if pval_s > alpha:
		print('Samples are uncorrelated (fail to reject H0) p=%.3f' % pval_s)
	else:
		print('Samples are correlated (reject H0) p=%.3f' % pval_s)

	# Plot regression line
	slope, intercept, r, p, stderr = linregress(X.iloc[:,0], y)
	line = f'Regression line: y={intercept:.2f}+{slope:.6f}x, r={r:.6f}'
	# plot the regression
	fig, ax = plt.subplots()
	ax.plot(X.iloc[:,0], y, linewidth=0, marker='s', label='Data points')
	ax.plot(X.iloc[:,0], intercept + slope * X.iloc[:,0], label=line)
	ax.set_xlabel(X.iloc[:,0].name)
	ax.set_ylabel(X.iloc[:,-1].name)
	ax.legend(facecolor='white')
	plt.show()

	# SelectKBest
	k_imp = 5
	fs = SelectKBest(score_func=f_regression, k=k_imp)
	# apply feature selection
	X_selected = fs.fit_transform(X, y)
	print(X_selected.shape)


	# PCA



	#SVR
	X_sc = StandardScaler().fit_transform(X)
	pdb.set_trace()
	y = np.array(y).reshape(-1,1)
	y_sc = StandardScaler().fit_transform(y)

	svr = SVR(kernel = 'rbf')
	model = svr.fit(X_sc,y_sc.ravel())

	pdb.set_trace()







