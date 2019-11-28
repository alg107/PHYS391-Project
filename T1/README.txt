- Main file is lf.py which generates samples from the LF and builds up a rough histogram
    - Saves these samples as a numpy array as well as samples from IMF to files so that
      these very long computations don't need to be repeated

- reconstruct.py is used to generate prettier pictures of the histogram to match up with
  the paper as well as fitting the models and combining the saved samples etc.

- fitting.py is called from reconstruct.py and is a wrapper of sorts around my non-linear
  fitting library AGModels as well as doing quite a lot of the plotting work too

- IMF.py is a small file that contains a couple things used in generating the IMF samples

- iso.db is the file containing all isochrone data
