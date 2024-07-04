# paper_biasSpeedAccel
Files containing the preprocessing and analysis done in the paper : Anticipatory smooth eye movements scale with the probability of visual motion: role of target speed and acceleration (Vanessa Carneiro Morita, David Souto, Guillaume S. Masson, Anna Montagnini)

Start with the preprocessing by running the files in the following order:
1. preprocessing/exp[XX]_preprocessing.py
2. preprocessing/exp[XX]_qualitycontrol.py
3. preprocessing/exp[XX]_postprocessing.py

After running these files, export the data to a csv by running the following file:
analysis/exp[XX}_exportANEMOparams.py

Run the LMM analysis on R:
analysis/exp[XX]_[condXX]_LMM.R

Plot the figures with:
plots/exp[XX]_[condXX]_plotParams.py

For the analysis of temporal window of integration, run:
analysis/exp2_temporalIntegrationWindows.py

