import numpy as np
from numpy.linalg import norm
from multipledispatch import dispatch
import plotly.graph_objects as go


# generic math functions
# https://stackoverflow.com/questions/5666222/3d-line-plane-intersectio
def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
            (v0[0] * v1[0]) +
            (v0[1] * v1[1]) +
            (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )


class Plane:
    def __init__(self, plane=[0, 0, 1, 0]):
        self.plane = plane
        self.a = plane[0]
        self.b = plane[1]
        self.c = plane[2]
        self.d = plane[3]
        self.normal = np.array([self.a, self.b, self.c])
        self.cx = None
        self.cy = None
        self.cz = None

        # init
        self.p_origin = self.closest_point_to_origin()


    def find_x(self, y, z):
        return (-self.d - self. b * y - self. c * z) / self.a

    def find_y(self, x, z):
        return (-self.d - self. a * x - self. c * z) / self.b

    def find_z(self, x, y):
        return (-self.d - self. a * x - self. b * y) / self.c


    def closest_point_to_origin(self):
        '''
        returns the closest point of a plane which is the closest to the origin <0, 0, 0>
        Stable method to find some arbitrary point in plane
        '''
        try:
            self.plane = np.asarray(self.plane).reshape(4)
        except ValueError as e:
            raise e
        n, d = self.plane[:3], - self.plane[-1]

        return d * (n / np.dot(n, n))

    # point plane intersection
    # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    # https://developer.blender.org/diffusion/B/browse/master/source/blender/blenlib/intern/math_geom.c;6856ea06425357921bd9a7e5089c2f2adbf1413a$1352
    def isect_line_plane_v3_4d(self, p0, p1, epsilon=1e-6):
        u = sub_v3v3(p1, p0)
        dot = dot_v3v3(self.plane, u)

        if abs(dot) > epsilon:
            # Calculate a point on the self.plane
            # (divide can be omitted for unit hessian-normal form).
            p_co = mul_v3_fl(self.plane, -self.plane[3] / len_squared_v3(self.plane))

            w = sub_v3v3(p0, p_co)
            fac = -dot_v3v3(self.plane, w) / dot
            u = mul_v3_fl(u, fac)
            return add_v3v3(p0, u)

        return None

    # point-normal plane
    def isect_line_plane_v3(self, p0, p1, p_co, p_no, epsilon=1e-6):
        u = p1 - p0
        dot = p_no * u
        if abs(dot) > epsilon:
            w = p0 - p_co
            fac = -(self.plane * w) / dot
            return p0 + (u * fac)

        return None

    # normal-form plane
    def isect_line_plane_v3_4d(self, p0, p1, epsilon=1e-6):
        u = p1 - p0
        dot = self.plane.xyz * u
        if abs(dot) > epsilon:
            p_co = self.plane.xyz * (-self.plane[3] / self.plane.xyz.length_squared)

            w = p0 - p_co
            fac = -(self.plane * w) / dot
            return p0 + (u * fac)

        return None


class Projection:
    def __init__(self, mesh, plane) -> None:
        self.mesh = mesh
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.areas = None
        self.plane = plane
        self.plane_normal = plane.normal

        # visualization
        # self.quick_fig = go.Figure()

        # init internals
        _ = self._update_plane_normal()
        self.calculate_triangle_area()

    def _update_plane_normal(self):
        self.plane_normal = np.array([self.plane.a, self.plane.b, self.plane.c])
        return self.plane_normal

    def calculate_triangle_area(self):
        # TODO: Multithreading

        self.areas = np.zeros((len(self.faces)))

        # we calculate areas of all faces in our mesh
        for i in range(len(self.faces)):
            # self.areas[i] = self.triangle_area(self.vertices[self.faces[i][0]],
            #                                 self.vertices[self.faces[i][1]],
            #                                 self.vertices[self.faces[i][2]])

            self.areas[i] = self.triangle_area(self.vertices[self.faces[i, :]])
        return self.areas

    def quick_fig(self, bounds=[10, 10, -10, -10]):

        volumes = self.volume_mesh_projected(self.vertices, self.faces)

        areas = self.calculate_triangle_area()

        norm = np.max(areas)
        areas_n = areas / norm

        # Plot Mesh surfaces
        fig = go.Figure(
            data=[go.Mesh3d(
                x=self.vertices[:, 0], y=self.vertices[:, 1], z=self.vertices[:, 2],
                i=self.faces[:, 0], j=self.faces[:, 1], k=self.faces[:, 2],
                colorbar_title='Volume',
                colorscale=[[0, 'gold'],
                            [0.5, 'mediumturquoise'],
                            [1, 'magenta']],
                # Intensity of each vertex, which will be interpolated and color-coded
                intensity=volumes,
                intensitymode='cell',
                # facecolor= areas,
                showscale=True
            )])

        # insert Vertices and Edges as polygon borders
        for i in range(len(self.faces)):
            xi = [self.vertices[xi, 0] for xi in self.faces[i]]
            yi = [self.vertices[xi, 1] for xi in self.faces[i]]
            zi = [self.vertices[xi, 2] for xi in self.faces[i]]
            xi.append(xi[0])
            yi.append(yi[0])
            zi.append(zi[0])
            fig.add_scatter3d(
                x=xi, y=yi, z=zi,
                marker=dict(
                    color='black',
                    symbol='square',
                    size=4),
                showlegend=False
            )

        pp, _ = self.project_points(self.vertices)

        fig.add_mesh3d(
            x=pp[:, 0], y=pp[:, 1], z=pp[:, 2],
            i=self.faces[:, 0], j=self.faces[:, 1], k=self.faces[:, 2],
            colorbar_title='Volume',
            colorscale=[[0, 'gold'],
                        [0.5, 'mediumturquoise'],
                        [1, 'magenta']],
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity=volumes,
            intensitymode='cell',
            # facecolor= areas,
            showscale=False
        )

        # Insert projection Plane
        plane, faces = self.projection_plane(self.plane, bounds)
        fig.add_mesh3d(x=plane[:, 0], y=plane[:, 1], z=plane[:, 2],
                       i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                       color='lightpink',
                       opacity=0.50
                       )
        return fig

    def project_points(self, points):
        p0 = self.plane.p_origin
        n = self.plane.normal

        normal = n / np.linalg.norm(n)  # ensure normal unit vector

        v = points - p0
        dist = np.dot(v, normal)  # Projection length
        point_projection = points - np.dot(dist.reshape(-1, 1), normal.reshape(1, -1))

        return point_projection, dist  # return projection length

    def projection_plane(self, plane, bounds):
        if plane.cx and plane.cy and plane.cz:
            center = [plane.cx, plane.cy, plane.cz]
        else:
            c = plane.p_origin

        xl, yl = bounds[0] - c[0], bounds[1] - c[1]
        xr, yr = bounds[2] - c[0], bounds[3] - c[1]

        z = lambda x, y: (-plane.d - plane.a * x - plane.b * y) / plane.c

        plane = np.array([
            [xl, yl, z(xl, yl)],
            [xr, yl, z(xr, yl)],
            [xr, yr, z(xr, yr)],
            [xl, yr, z(xl, yr)],
            # [xl, yl, z(xl, yl)],  # close polygon
        ])
        faces = np.array([[0, 1, 2], [2, 3, 0]])

        return plane, faces

    # Plot plane by 4 Points
    def plot_plane(self, points, plane=[0, 0, 1, 0], bounds=[10, 10, -10, -10]):

        plane, faces = self.projection_plane(plane, bounds)

        fig = go.Figure(
            data=[go.Mesh3d(x=plane[:, 0], y=plane[:, 1], z=plane[:, 2],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            color='lightpink',
                            opacity=0.50
                            )])

        pp, _ = self.project_points(points)

        fig.add_scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers')

        for i in range(points.shape[0]):
            x = [points[i, 0], pp[i, 0]]
            y = [points[i, 1], pp[i, 1]]
            z = [points[i, 2], pp[i, 2]]

            fig.add_scatter3d(x=x, y=y, z=z
                              )

        fig.add_scatter3d(x=pp[:, 0], y=pp[:, 1], z=pp[:, 2],
                          mode='markers',
                          marker=dict(
                              color='black',
                              symbol='square',
                              size=4), )

        return fig

    @dispatch(np.ndarray)
    def triangle_area(points):
        points = np.asarray(points).reshape(3, 3)
        # function to calculate trinagle area by its verticies
        # https://en.wikipedia.org/wiki/Heron%27s_formula

        sides = np.array([
            norm(points[0] - points[1]),
            norm(points[1] - points[2]),
            norm(points[2] - points[0]),
        ])

        s = 0.5 * (sides[0] + sides[1] + sides[2])
        return max(s * (s - sides[0]) * (s - sides[1]) * (s - sides[2]), 0) ** 0.5

    @dispatch(np.ndarray, np.ndarray, np.ndarray)
    def triangle_area(pt1, pt2, pt3):
        # function to calculate trinagle area by its verticies
        # https://en.wikipedia.org/wiki/Heron%27s_formula
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def triangle_center_of_gravity(self, points):
        n_points = points.shape[0]
        s_xyz = np.sum(points, axis=0)  # sum of points <sum_x, sum_y, sum_z>
        return s_xyz / n_points

    def volume_wedge(self, vertices):
        cog = self.triangle_center_of_gravity(vertices)

        pp_cog, dist = self.project_points(cog)
        pp_vertices, _ = self.project_points(vertices)

        pp_area = self.triangle_area(pp_vertices)

        return pp_area * dist

    def volume_mesh_projected(self, vertices, faces):

        n_faces = faces.shape[0]
        areas = np.zeros([n_faces])

        for i in range(n_faces):
            areas[i] = self.volume_wedge(vertices[faces[i], :])

        return areas



