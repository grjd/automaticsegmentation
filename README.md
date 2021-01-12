# Dataset supplement of volume estimates of subcortical strctures with FSL and FreeSurfer
This repository contains code and data used in the following publication:

J. Gomez-Ramirez et al, "A comparative analysis of automated MRI brain segmentation in a large longitudinal dataset of elderly subjects" (pre-print on BioRxiv: https://doi.org/10.1101/2020.08.13.249474 )

**Abstract***

In this study, we perform a comparative analysis of automated segmentation of subcortical structures in the elderly brain. Manual segmentation is an extremely time-consuming task and automated methods are thus, gaining importance as clinical tool for diagnosis. In the last few years, AI-based segmentation has delivered in some cases superior results than manual segmentation, in both time and accuracy. 
To test the performance of automated segmentation methods, the two most commonly used software libraries for brain segmentation -FreeSurfer and FSL- are put to work in a large dataset of 4000 MRI data collected for this study.
We find a lack of linear correlation between the segmentation volume estimates obtained from FreeSurfer and FSL. On the other hand, FreeSurfer volume estimates tend to be larger than FSL estimates of the areas putamen, thalamus, amygdala, caudate, pallidum, hippocampus and accumbens.
In the era of big data, automated segmentation is called to play a preponderant role in medical imaging. The characterization of the performance of brain segmentation algorithms in large datasets as the one presented here, is a matter of scientific and clinical interest now and for the immediate future. 

Dataset description

The dataset contains two csv files, *df_fsl_lon.csv* is the Pandas dataframe containing the the results of the automted segmentation with FSL and *df_free_lon.csv* contained the automated segmentation output obtained with FreeSurfer. 
The fields include in the dataset are as follows:
- Age the age of the participant in the moemnt of performing the MRI scan (%.2f)
- Sex encoded as 0 Male and 1 Female
- Subcortical Volume estimates use the nomenclature:, [fsl|free]_[R|L]|[strcture] where structure can be Thalamus, Accumbens, Pallidum, Hippocampus, Amygdala, Caudate and Putamen. The volume is expressed in  $$mm^3$$.

```
df_fs_lon.csv.shape
[7080 rows x 16 columns]
df_fsl_lon.columns
Index(['age', 'sex', 'fsl_R_Thal', 'fsl_L_Thal', 'fsl_R_Puta', 'fsl_L_Puta',
       'fsl_R_Amyg', 'fsl_L_Amyg', 'fsl_R_Pall', 'fsl_L_Pall', 'fsl_R_Caud',
       'fsl_L_Caud', 'fsl_R_Hipp', 'fsl_L_Hipp', 'fsl_R_Accu', 'fsl_L_Accu'],
      dtype='object')
      
df_free_lon.shape
[7080 rows x 16 columns] 
df_free_lon.columns
Index(['age', 'sex', 'free_R_Thal', 'free_L_Thal', 'free_R_Puta',
       'free_L_Puta', 'free_R_Amyg', 'free_L_Amyg', 'free_R_Pall',
       'free_L_Pall', 'free_R_Caud', 'free_L_Caud', 'free_R_Hipp',
       'free_L_Hipp', 'free_R_Accu', 'free_L_Accu'],
      dtype='object')
```   
