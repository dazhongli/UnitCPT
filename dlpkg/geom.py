from numpy.lib.npyio import load
import pandas as pd
import numpy as np
import geoplot as plt
from tqdm import tqdm
import plotly.graph_objects as go


class Joint():
    def __init__(self):
        self.plane_axis = None

    def construct_plane_axis(self, chord, brace):
        cx = chord.axis[:, 0]
        bx = brace.axis[:, 0]
        plane_x = self.normalise_vector(np.cross(cx, bx))
        plane_y = chord.axis[:, 0]
        plane_z = self.normalise_vector(np.cross(plane_x, plane_y))
        self.plane_axis = np.array([plane_x, plane_y, plane_z]).transpose()
        return self.plane_axis

    def normalise_vector(self, v):
        return v/np.linalg.norm(v)

    def angle(self, elem1, elem2):
        elem1_dir = elem1.direction()
        elem2_dir = elem2.direction()
        angle = np.degrees(np.arccos(elem1_dir@elem2_dir))
        if angle >= 90:
            return 180-angle
        else:
            return angle


class Node():
    def __init__(self, node_id, df_coord):
        self.id = node_id
        node_record = df_coord.loc[node_id, :]
        self.coord = [node_record.x, node_record.y, node_record.z]


class Elem():
    def __init__(self, id, node1, node2, df_coord, df_axes):
        self.id = id
        self.n1 = Node(node1, df_coord)
        self.n2 = Node(node2, df_coord)
        axis = np.array(df_axes.loc[id, :]).reshape(3, 3)
        self.axis = axis.transpose()  # we would like to
        self.results = {}

    def add_results(self, load_case, df_results):
        # get the results for the element first
        df_elem = df_results[df_results.Elem == self.id].copy()
        df_elem = df_elem.set_index('Pos')
        self.results[load_case] = df_elem

    def clear_results(self):
        self.results = {}

    def get_force(self, load_case, node_id, new_axis=None):
        '''
        Return the bending moment of the element at a particular node 
        ['Mxx','Myy','Mzz']
        '''
        if new_axis is None:
            Mxx, Myy, Mzz = np.array(
                self.results[load_case].loc[node_id, ['Mxx', 'Myy', 'Mzz']])
            Fx = self.get_axial_force(load_case, node_id)
            df = pd.DataFrame(dict(Fx=[Fx], Mxx=[Mxx], Myy=[Myy], Mzz=[Mzz]))
            return df

        else:
            bm_local = np.array(
                self.results[load_case].loc[node_id, ['Mxx', 'Myy', 'Mzz']])
            bm_new_plane = np.linalg.inv(new_axis)@self.axis@bm_local
            axial_force = self.get_axial_force(load_case, node_id)
            mi, mo, _ = bm_new_plane
            df = pd.DataFrame(dict(Fx=[axial_force], M_i=[mi], M_o=[mo]))
            return df

    def get_axial_force(self, load_case, node_id):
        '''
        Return the axial force of the element
        '''
        return self.results[load_case].loc[node_id, 'Fx']

    def direction(self):
        return self.axis[:, 0]

    def relative_angle(self, to_elem):
        elem1_dir = to_elem.direction()
        elem2_dir = self.direction()
        angle = np.degrees(np.arccos(elem1_dir@elem2_dir))
        if angle >= 90:
            return 180-angle
        else:
            return angle


class Mesh():
    def __init__(self, df_topo, df_coord, df_axes):
        self.elems = {}

        for ix, row in df_topo.iterrows():
            self.elems[ix] = Elem(ix, row.N1, row.N2, df_coord, df_axes)

        self.nodes = {}
        for ix, row in df_coord.iterrows():
            self.nodes[ix] = Node(ix, df_coord)

    def attach_results(self, load_case, df_results):
        for _, elem in self.elems.items():
            elem.add_results(load_case, df_results)
        # for ix, row in df_results.iterrows():
        #     self.elems[ix].add_results(load_case, df_results)

    def plot(self):
        fig = plt.GEOPlot.get_figure()
        for key, elem in tqdm(self.elems.items()):
            node1 = elem.n1.coord
            node2 = elem.n2.coord
            points = np.array([node1, node2])
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='lines', line=dict(color='black'), showlegend=False))
        self.fig = fig
        return fig
