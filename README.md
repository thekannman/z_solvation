This program is intended to mark solvent molecules that are within a particular solvation shell of a given 
solute group.

THIS PROGRAM IS NOT THOROUGHLY TESTED!!!

Current limitations include:
* The input trajectory must be in the .xtc format used by the GROMACS simulation package.
* Only individual atoms are currently evaluated, not molecular centers of mass.

The following libraries are required by the program:
* GROMACS XTC library for reading position/velocity files. 
  http://www.gromacs.org/Developer_Zone/Programming_Guide/XTC_Library
* The Armadillo library for matrix calculations. 
  http://arma.sourceforge.net/
* The Boost program_options library
  http://www.boost.org/
