# PCR_Blocker

A Python code developed in 
Takahashi T, Aoyanagi H, Pigolotti S, Toyabe S
Blocking uncertain mispriming errors of PCR (2024)
https://www.biorxiv.org/content/10.1101/2024.04.19.590219v1

This program is used for the blocking (clamping) method of PCR to suppress the mispriming errors.
This program finds the optimal blocker concentrations when we add a combination of PCR blockers for a given template and contaminating sequences.

Please take a look at the comments in the code for details.

The parameters are specific to the PCR condition used in the paper (Taq DNA polymerase (New England Biolabs) was used).
Tuning might be necessary for other conditions.

PCRblocker.py contains the main functions.
optimization.py demonstrates the usage.

You need [BioPhython](https://biopython.org/) library for the execution.

Contact:

Shoichi Toyabe (Tohoku University, Japan)

toyabe@tohoku.ac.jp
