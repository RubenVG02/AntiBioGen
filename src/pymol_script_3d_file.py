# Description: This script is used to generate a pymol script to generate 3D
# representations of molecules from a smile sequence.
#

import cmd

def pymol_script_3d_file(smile, name_load, name_save):
    '''
    Function to generate a pymol script to generate 3D representations of molecules from a smile sequence.

    Parameters:
        -smile: smile sequence from which to obtain the sdf file
        -name_load: name of the sdf file to be created
        -name_save: name of the stl file to be created
    
    
    '''
    cmd.load(f"{name_load}.sdf")
    cmd.save(f"{name_save}.stl")
    cmd.quit()
    print("The stl file has been created")

pymol_script_3d_file("CC(C)(C)OC(=O)N1CCC(CC1)Nc2nc3ccccc3s2", "molecule_sdf", "molecule_stl")