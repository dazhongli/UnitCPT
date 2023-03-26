from UnitCPT import gpd, CPT, Path, pd
from UnitCPT.proj import PROJ_DATA


def read_proj_coords(project_path, project_name):
    '''
    Read the coordinates of the SI and return a gdf
    The file should be shapefile
    '''
    proj_path = Path(project_path)
    filename = proj_path / project_name / 'data' / 'shp' / 'CPT_coords.json'
    assert (filename.exists())
    gdf = gpd.read_file(filename)
    return gdf


def get_cpt(filename, PROJ_DATA: dict) -> CPT:
    '''
    Get a CPT object from a string
    '''
    cpt_path = Path(PROJ_DATA['proj_path']) / \
        PROJ_DATA['active_project'] / 'CPT'/filename
    df = pd.read_json(cpt_path)
    cpt = CPT()
    cpt.df = df
    return cpt
