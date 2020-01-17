# Running this code

## Luminosity Function (LF)

Enter the LF folder and run
```bash
python MonteCarlo.py
```
to generate samples using a Monte Carlo method. Next, run
```bash
python SALF.py
```
to generate a luminosity function using the semi-analytic method. 
Finally, run
```bash
python FormatLF.py
```
to present this data nicely and give a comparison of the two.
You should just be able to run this last command as data from when I've
run the first two commands will most likely still be present.

## VVV Data Processing

I will update this on how to generate the data used but for now you
can just navigate into the VVV folder and run
```bash
python display.py
```
