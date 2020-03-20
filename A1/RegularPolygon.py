from math import sin, cos, tan, pi
class RegularPolygon:
    def __init__(self, num_sides, bbox_side, x, y):
        self.center_x = x
        self.center_y = y
        self.bbox_side = bbox_side
        self.num_sides = num_sides
        self.apothem = bbox_side / 2
        self.side_length = 2 * self.apothem * tan(pi / 6)
        self._make_points(x, y)

    def _make_points(self, center_x, center_y):
        self.point_list = []
        _angle = 2 * pi / self.num_sides
        for pdx in range(self.num_sides):
            angle = _angle * pdx - pi / 2
            _x = cos(angle) * self.side_length + center_x
            _y = sin(angle) * self.side_length + center_y
            self.point_list.append(_x)
            self.point_list.append(_y)