# Running this code

I would recommend using some kind of python environment 
manager such as Anaconda or Virtual Environment. There are a few
dependencies. Mostly things like Numpy, Scipy, Matplotlib etc. but there
are a few other ones. I'll compile a dependencies list at some point.

## Stellar Luminosity Function Construction

To generate samples using a Monte Carlo method, enter the LF folder and run
```bash
python MonteCarlo.py
```
Next, to generate a luminosity function using the semi-analytic method run
```bash
python SALF.py
```
Finally, to present this data nicely and give a comparison of the two methods run
```bash
python FormatLF.py
```
You should just be able to run this last command as data from when I've
run the first two commands will most likely still be present.

## VVV Data Processing (Not included in report)

I will update this on how to generate the data used but for now you
can just navigate into the VVV folder and run
```bash
python display.py
```
