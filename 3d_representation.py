import requests

def get_sdf(smile, name):
    '''
    Function to obtain the sdf file from a smile sequence in order to be able to represent it in 3D

    Parameters:
        -smile: smile sequence from which to obtain the sdf file
        -name: name of the sdf file to be created

    Returns:
        -sdf_file: sdf file obtained from the smile sequence
    '''
    request=requests.get(f"https://cactus.nci.nih.gov/chemical/structure/{smile}/file?format=sdf&get3d=True")
    sdf_file=request.text
    print(sdf_file)
    if "Page not found" in sdf_file:
        print("SDF file not found")
        return "SDF file not found"
    else:
        with open(f"{name}.sdf","w") as f:
            f.write(sdf_file)
        print("SDF file created")
    
    

### 3D REPRESENTATION ###

#All these lines need to be run using the PyMol console 

'''


# 1. Open PyMol
open -a PyMOL

# 2. Load the sdf file
cmd.load('molecule_sdf.sdf')

# 3. Save the molecule in stl format
cmd.save('molecule_stl.stl')
    


'''