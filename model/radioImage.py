from plotly.graph_objs import Layout

from model.ellipsoid import Ellipsoid
from model.ray import Ray
import numpy.linalg as lg
import plotly.graph_objects as pgo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as trim
from scipy.spatial import Delaunay, delaunay_plot_2d
import plotly as pl
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import matplotlib.cm as cm
import alphashape
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import math
from matplotlib.collections import LineCollection
from shapely.geometry import MultiLineString
from shapely.ops import cascaded_union, polygonize
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class RadioImage:
    def __init__(self, ell: Ellipsoid):
        self.ell = ell
        self.rays = []
        self.distances = []
        self.velocities = []
        self.colors = []
    
    def add_ray(self, ray: Ray):
        self.rays.append(ray)
        self.distances.append(ray.distance_to_point)
        self.velocities.append(ray.velocity_sign_direction * lg.norm(ray.velocity_projection))
        self.colors.append(ray.color)
    
    def build_image(self, name, x_lim, y_lim):
        self.build_image_interp(name, x_lim, y_lim)
        
        # self.build_image_points()
    
    def build_image_interp(self, name, x_lim, y_lim):
        x = np.array(self.velocities)
        y = np.array(self.distances)
        z = np.array(self.colors)
        z = z / z.max()
        
        points = np.column_stack((x, y))
        triangulation = trim.Triangulation(x, y)
        
        edges = set()
        edge_points = []
        
        alpha_nose = 10
        alpha_side = 0.0000001
        
        delta_y = y.max() - y.min()
        x_min = -0.5
        x_max = 0.35
        y_min = y.min()
        y_max = y.min() + delta_y / 5
        
        def add_edge(i, j):
            if (i, j) in edges or (j, i) in edges:
                return
            edges.add((i, j))
            edge_points.append(points[[i, j]])
        
        def get_plot_part(a, b, c):
            for p in [a, b, c]:
                if x_min < p[0] < x_max and y_min < p[1] < y_max:
                    return alpha_nose
            
            return alpha_side
        
        def get_center_of_tri(tri):
            sum_x = 0
            sum_y = 0
            
            for i in tri:
                sum_x += points[i][0]
                sum_y += points[i][1]
            
            x_c = round(sum_x / 3, 2)
            y_c = round(sum_y / 3, 2)
            
            return np.array((x_c, y_c))
        
        def point_in_alpha_shape(boundaries, point):
            tuple_shape = list(zip(boundaries.xy[0], boundaries.xy[1]))
            point = Point(point)
            polygon = Polygon(tuple_shape)
            
            return polygon.contains(point)
        
        for ia, ib, ic in triangulation.triangles:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            
            # Lengths of sides of triangle
            a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
            
            # Semiperimeter of triangle
            s = (a + b + c) / 2.0
            
            # Area of triangle by Heron's formula
            area = math.sqrt(s * (s - a) * (s - b) * (s - c))
            
            circum_r = a * b * c / (4.0 * area)
            
            alpha = get_plot_part(pa, pb, pc)
            # Here's the radius filter.
            if circum_r < 1.0 / alpha:
                add_edge(ia, ib)
                add_edge(ib, ic)
                add_edge(ic, ia)
        
        # lines = LineCollection(edge_points)
        # plt.figure()
        # plt.title('Alpha=2.0 Delaunay triangulation')
        # plt.gca().add_collection(lines)
        # plt.plot(points[:, 0], points[:, 1], 'o')
        
        m = MultiLineString(edge_points)
        triangles = list(polygonize(m))
        
        boundaries = cascaded_union(triangles).boundary
        # x_alpha = boundaries.xy[0]
        # y_alpha = boundaries.xy[1]
        
        out_of_alpha = []
        inside_alpha = []
        valid_tri = []
        
        for tri in triangulation.triangles:
            center = get_center_of_tri(tri)
            
            if point_in_alpha_shape(boundaries, center):
                inside_alpha.append(center)
                valid_tri.append(tri)
            else:
                out_of_alpha.append(center)
        
        out_of_alpha = np.array(out_of_alpha).T
        inside_alpha = np.array(inside_alpha).T
        
        valid_triangulation = trim.Triangulation(x, y, valid_tri)

        fig, ax = plt.subplots()
        ax.set(xlim=x_lim, ylim=y_lim)
        ax.ticklabel_format(useOffset=False)
        plt.axis('off')
        plt.gca().set_aspect('equal')
        fig.patch.set_facecolor('black')

        # plt.gca().add_patch(PolygonPatch(cascaded_union(triangles), alpha=0.5))
        # plt.scatter(x_alpha, y_alpha, marker="*", s=20)
        
        # plt.scatter(out_of_alpha[0], out_of_alpha[1], marker="*", s=20, c='blue')
        
        # xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000))
        # interp_cubic_geom = trim.CubicTriInterpolator(valid_triangulation, z, kind='geom')
        # zi_cubic_geom = interp_cubic_geom(xi, yi)
        # cs = ax.contourf(xi, yi, zi_cubic_geom, 300, cmap="Greys", vmin=0, vmax=80)
        # fig.colorbar(cs, ax=ax)
        
        color_map = plt.cm.get_cmap('Greys')
        plt.tricontourf(valid_triangulation, z, 400, cmap=color_map, vmin=0, vmax=z.max())
        # plt.scatter(0, np.linalg.norm(self.ell.x), marker="o", s=10, c='red')

        # plt.scatter(x, y, marker="o", s=10, c='red')
        # plt.vlines(x_min, y_min, y_max)
        # plt.vlines(x_max, y_min, y_max)
        # plt.hlines(y_min, x_min, x_max)
        # plt.hlines(y_max, x_min, x_max)
        
        # plt.show()
        plt.savefig(name, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    def build_image_points(self):
        x = np.array(self.velocities)
        y = np.array(self.distances)
        z = np.array(self.colors)
        
        layout = pgo.Layout(
            paper_bgcolor='rgba(0,0,0,1)',
            plot_bgcolor='rgba(0,0,0,1)'
        )
        
        radio_scatter = pgo.Figure(data=pgo.Scatter(x=x,
                                                    y=y,
                                                    mode='markers',
                                                    marker=dict(
                                                        size=10,
                                                        color=z / z.max(),
                                                        colorscale='gray',
                                                        showscale=True,
                                                        reversescale=True
                                                    )
                                                    )
                                   , layout=layout
                                   )
        z = np.array(self.colors)
        print(f"\n{z.max()}")
        
        radio_scatter.show()
        radio_scatter.write_html('radio_image.html', auto_open=True)
