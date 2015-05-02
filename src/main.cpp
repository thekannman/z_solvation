//Copyright (c) 2015 Zachary Kann
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

// ---
// Author: Zachary Kann

#include "xdrfile_trr.h"
#include "boost/program_options.hpp"
#include "z_sim_params.hpp"
#include "z_vec.hpp"
#include "z_file.hpp"
#include "z_conversions.hpp"
#include "z_molecule.hpp"
#include "z_atom_group.hpp"
#include "z_gromacs.hpp"

namespace po = boost::program_options;
// Units are nm, ps.

int main (int argc, char *argv[]) {
  int st;
  SimParams params;

  po::options_description desc("Options");
  desc.add_options()
    ("help,h",  "Print help messages")
    ("group,g", po::value<std::string>()->default_value("He"),
     "Name of solute group")
    ("solvent,s", po::value<std::string>()->default_value("OW"),
     "Name of solvent group")
    ("rcut1,r",
     po::value<double>()->default_value(0.0),
     "Cutoff radius for 1st solvation shell")
    ("rcut2,r",
     po::value<double>()->default_value(0.0),
     "Cutoff radius for 2nd solvation shell")
    ("rcut3,r",
     po::value<double>()->default_value(0.0),
     "Cutoff radius for 3rd solvation shell")
    ("index,n", po::value<std::string>()->default_value("index.ndx"),
     ".ndx file containing atomic indices for groups")
    ("gro", po::value<std::string>()->default_value("conf.gro"),
     ".gro file containing list of atoms/molecules")
    ("top", po::value<std::string>()->default_value("topol.top"),
     ".top file containing atomic/molecular properties")
    ("max_time,t",
     po::value<double>()->default_value(0.0),
     "Maximum simulation time to use in calculations");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(EXIT_SUCCESS);
  }

  std::map<std::string, std::vector<int> > groups;
  groups = ReadNdx(vm["index"].as<std::string>());
  std::vector<Molecule> molecules = GenMolecules(vm["top"].as<std::string>(),
                                                 params);
  AtomGroup all_atoms(vm["gro"].as<std::string>(), molecules);
  AtomGroup solute_group(vm["group"].as<std::string>(),
                           SelectGroup(groups, vm["group"].as<std::string>()),
                           all_atoms);
  AtomGroup solvent_group(vm["solvent"].as<std::string>(),
                         SelectGroup(groups, vm["solvent"].as<std::string>()),
                         all_atoms);

  rvec *x_in = NULL;
  matrix box_mat;
  arma::rowvec box = arma::zeros<arma::rowvec>(DIMS);
  std::string xtc_filename = "prod.xtc";
  std::string trr_filename = "prod.trr";
  XDRFILE *xtc_file;
  params.ExtractTrajMetadata(strdup(xtc_filename.c_str()), (&x_in), box);
  xtc_file = xdrfile_open(strdup(xtc_filename.c_str()), "r");
  params.set_box(box);
  params.set_max_time(vm["max_time"].as<double>());

  const double kMinimumDistanceSquared = 0.001*0.001;
  const double rcut1_squared =
      vm["rcut1"].as<double>()*vm["rcut1"].as<double>();
  const double rcut2_squared =
      vm["rcut2"].as<double>()*vm["rcut2"].as<double>();
  const double rcut3_squared =
      vm["rcut3"].as<double>()*vm["rcut3"].as<double>();
  bool calculate_1st_shell =
      (rcut1_squared > kMinimumDistanceSquared) ? true : false;
  bool calculate_2nd_shell =
      (rcut1_squared > kMinimumDistanceSquared) ? true : false;
  bool calculate_3rd_shell =
      (rcut1_squared > kMinimumDistanceSquared) ? true : false;
  assert(calculate_1st_shell || calculate_2nd_shell ||
         calculate_3rd_shell);

  std::string solv_1st_filename = vm["solvent"].as<std::string>() +
      "_within_1st_of_" + vm["group"].as<std::string>() + "_.dat";
  std::string solv_2nd_filename = vm["solvent"].as<std::string>() +
      "_within_2nd_of_" + vm["group"].as<std::string>() + "_.dat";
  std::string solv_3rd_filename = vm["solvent"].as<std::string>() +
      "_within_3rd_of_" + vm["group"].as<std::string>() + "_.dat";

  std::ofstream solv_1st;
  std::ofstream solv_2nd;
  std::ofstream solv_3rd;
  if (calculate_1st_shell)
    solv_1st.open(solv_1st_filename.c_str());
  if (calculate_2nd_shell)
    solv_2nd.open(solv_2nd_filename.c_str());
  if (calculate_3rd_shell)
    solv_3rd.open(solv_3rd_filename.c_str());

  solv_1st.width(0);
  solv_2nd.width(0);
  solv_3rd.width(0);

  arma::irowvec within_1st = arma::zeros<arma::irowvec>(solvent_group.size());
  arma::irowvec within_2nd = arma::zeros<arma::irowvec>(solvent_group.size());
  arma::irowvec within_3rd = arma::zeros<arma::irowvec>(solvent_group.size());

  arma::rowvec dx;
  float time, prec;
  int step = 0;
  for (step = 0; step < params.max_steps(); ++step) {
    if(read_xtc(xtc_file, params.num_atoms(), &st, &time, box_mat, x_in, &prec))
      break;
    params.set_box(box_mat);
    int i = 0;
    for (std::vector<int>::iterator i_atom = solute_group.begin();
         i_atom != solute_group.end(); ++i_atom, ++i) {
      solute_group.set_position(i, x_in[*i_atom]);
    }
    i = 0;
    for (std::vector<int>::iterator i_atom = solvent_group.begin();
         i_atom != solvent_group.end(); ++i_atom, ++i) {
      solvent_group.set_position(i, x_in[*i_atom]);
    }
    within_1st.zeros();
    within_2nd.zeros();
    within_3rd.zeros();
    for (int i_solvent = 0; i_solvent < solvent_group.size(); ++i_solvent) {
      for (int i_solute = 0; i_solute < solute_group.size(); ++i_solute) {
        FindDxNoShift(dx,  solvent_group.position(i_solvent),
                      solute_group.position(i_solute), box);
        double r2 = arma::dot(dx, dx);
        if (r2 < rcut3_squared)
          within_3rd(i_solvent) = 1;
        if (r2 < rcut2_squared)
          within_2nd(i_solvent) = 1;
        if (r2 < rcut1_squared) {
          within_1st(i_solvent) = 1;
          break;
        }
      }
    }
    if (calculate_1st_shell)
      within_1st.raw_print(solv_1st);
      //WriteInt(files.solv, within, files.solvFile, 2, params.numMols);
    if (calculate_2nd_shell)
      within_2nd.raw_print(solv_2nd);
      //WriteInt(files.solv2, within2, files.solv2File, 2, params.numMols);
    if (calculate_3rd_shell)
      within_3rd.raw_print(solv_3rd);
      //WriteInt(files.solv3, within3, files.solv3File, 2, params.numMols);
  }
  solv_1st.close();
  solv_2nd.close();
  solv_3rd.close();
} // main
