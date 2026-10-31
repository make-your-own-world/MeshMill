import numpy as np

from constrained_solid import _nonmanifold_vertex_count


def test_closed_tetrahedron_has_one_vertex_fan():
    faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
    assert _nonmanifold_vertex_count(faces)==0

def test_point_touching_solids_are_rejected():
    faces=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2],[0,4,5],[0,6,4],[0,5,6],[4,6,5]])
    assert _nonmanifold_vertex_count(faces)==1
