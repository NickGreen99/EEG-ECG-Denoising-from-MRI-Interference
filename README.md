# Deep Residual Learning for Artifact Suppression in Sodium MRI–EEG

*Code for the ISMRM abstract “Deep Residual Learning for Artifact Suppression in Simultaneous Sodium MRI–EEG Acquisition.”* 

## Overview

Simultaneous sodium MRI (²³Na) and EEG provides powerful complementary information but suffers from severe MRI-induced gradient artifacts.  
This repository implements and compares three artifact-removal methods:

- **Independent Component Analysis (ICA)**
- **Residual U-Net** (deep encoder–decoder with residual + skip connections)
- **Variational Autoencoder (VAE)**

The **Residual U-Net** demonstrates the strongest artifact suppression and best preservation of physiologic EEG/ECG structure.

## Key Results

- **54% reduction** in ECG amplitude RMSE  
- **26% reduction** in z-scored RMSE  
- **3× increase** in Pearson correlation  
- ICA leaves residual high-frequency artifacts  
- VAE smooths spectra but loses some physiologic detail  

## Contact
Nikolaos.Prasinos@nyulangone.org

