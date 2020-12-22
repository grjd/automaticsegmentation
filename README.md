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
	df_fsl_lon, df_free_lon = df_fsl_lon.dropna(), df_free_lon.dropna()
	df_fsl_lon = outlier_detection_isoforest(df_fsl_lon)
	df_free_lon = outlier_detection_isoforest(df_free_lon)
```
