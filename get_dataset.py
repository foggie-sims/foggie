
import yt 

def get_dataset(enzo_output):

    # open the enzo output 
    ds = yt.load(enzo_output)
    return ds
