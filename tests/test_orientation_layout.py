import numpy as np

from orientation_layout import place_axis_badges


def test_rotating_badges_remain_clear_and_move_continuously():
    previous = None
    for angle in np.linspace(0, 2*np.pi, 1441):
        desired = [(95+67*np.cos(angle+i*2*np.pi/3),
                    95+67*np.sin(angle+i*2*np.pi/3)) for i in range(3)]
        points = place_axis_badges(desired, (55,55,135,135),190,190,30,34)
        for x,y in points:
            assert x-15 >= 3.99 and x+15 <= 186.01
            assert y-17 >= 3.99 and y+17 <= 186.01
            assert x+15 <= 49 or x-15 >= 141 or y+17 <= 49 or y-17 >= 141
        for i in range(3):
            for j in range(i):
                delta = np.abs(points[i]-points[j])
                assert delta[0] >= 34 or delta[1] >= 38
        if previous is not None:
            assert np.max(np.linalg.norm(points-previous, axis=1)) < 1
        previous = points
