# automaticsegmentation
Automatic Segmentation subcortical brain structures

Code for paper title.
Dataset description

Code Here
```python
	fsl_lon_cols = ['fsl_R_Thal', 'fsl_L_Thal', 'fsl_R_Puta', 'fsl_L_Puta','fsl_R_Amyg', 'fsl_L_Amyg', 'fsl_R_Pall', 'fsl_L_Pall', 'fsl_R_Caud','fsl_L_Caud', 'fsl_R_Hipp', 'fsl_L_Hipp', 'fsl_R_Accu', 'fsl_L_Accu']
	free_lon_cols = ['free_R_Thal', 'free_L_Thal', 'free_R_Puta','free_L_Puta', 'free_R_Amyg', 'free_L_Amyg', 'free_R_Pall','free_L_Pall', 'free_R_Caud', 'free_L_Caud', 'free_R_Hipp','free_L_Hipp', 'free_R_Accu', 'free_L_Accu']
	# Get longitudinal dataframe
	#oneway_anovatest(dataframe_orig)
	df_fsl_lon, df_free_lon = convertdf_intolongitudinal(dataframe_orig)
	# Remove NaNS
	df_fsl_lon, df_free_lon = df_fsl_lon.dropna(), df_free_lon.dropna()
	df_fsl_lon = outlier_detection_isoforest(df_fsl_lon)
	# IsoForest enesemble algorithm for outlier removal with contaminatio parameter (0,1) 
	df_free_lon = outlier_detection_isoforest(df_free_lon)
```
df_fsl_lon.shape == df_free_lon.shape (7000,16)
7000 rows and 16 columns, sex, age and 14 segemnatation estimates both hemispheres:
1. Thalamus
2. Putamen 
3. Amygdala 
4. Pallidum 
5. Caudate 
6. Hippocampus 
7. Accumbens
After remving the NaNs FSL lon 4068 rows and FreeSurfer 4009 rows.

eigenvalues analysis
https://math.stackexchange.com/questions/243533/how-to-intuitively-understand-eigenvalue-and-eigenvector
Whenall correlations are positive, this first eigenvalue is approximately a linear function of the average correlation among the variables.
The first eigenvalue \lambda_i indicates themaximum amount of the variance of the variables which can be ac-counted for with a linear model by a single underlying factor. All component analysis does is map the n(n - 1)/2 correlations among n variables into n eigenvalues and their associated eigenvectors, so the eigenvalues must be functions of those underlying correlations.
Discovering how the eigenvalues relate to the correlations would in-crease our intuitive understanding of them and of component analysismore generally.
https://journals.sagepub.com/doi/pdf/10.1177/001316448104100102

http://www.ccgalberta.com/ccgresources/report14/2012-408_understanding_correlation_matrices.pdf
