import yt

def get_proper_box_size(ds):
    return ds.get_parameter('CosmologyComovingBoxSize') / ((1 + ds.current_redshift) * ds.get_parameter('CosmologyHubbleConstantNow')) * 1000. # in kpc
