# FMR-analysis
This repo is for analysing FMR data taken at MIT. The code load the data from the text files with frequency in name in form of fXY and fit Lorentzian function. Outputs are linewidth, amplitude and position of the resonance

You can use 4 different models - standard/assymetric Lorentz funciton and its derivative. For **lock-in technique use derivative function**. I suggest to use assymteric as this should improve the fitting.

You can load several files at once - select any number of files you wish. **Name your file like this - fXXGHz-YYYYY.txt**. This will ensure correct export to csv/clipboard later.

If there are strange drifts or another resonances in which you are not interested of anylizing **mask them with available tool**

**Keep in mind correct units** (unfortunatelly setup saves the field in Oe and not T/mT).

Once the fitting is finished you can use _copy to clipboard tool_ or _save to csv_ located on the right panel of GUI.

In case of any question don't hesitate to aks!
