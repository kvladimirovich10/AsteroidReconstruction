from plotly.graph_objs import Layout

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
    def __init__(self):
        self.rays = []
        self.distances = []
        self.velocities = []
        self.colors = []
    
    def build_image(self):
        x = np.array(self.velocities)
        y = np.array(self.distances)
        z = np.array(self.colors)
        
        points = np.column_stack((x, y))
        triangulation = trim.Triangulation(x, y)  # Delaunay(points)
        
        edges = set()
        edge_points = []
        
        alpha_nose = 10
        alpha_side = 0.000000001
        
        delta_y = y.max() - y.min()
        x_min = -0.02
        x_max = 2.3
        y_min = y.min()
        y_max = y.min() + 3 * delta_y / 4
        
        def add_edge(i, j):
            """Add a line between the i-th and j-th points, if not in the list already"""
            if (i, j) in edges or (j, i) in edges:
                # already added
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
        x_alpha = boundaries.xy[0]
        y_alpha = boundaries.xy[1]
        
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
        ax.set_facecolor((0, 0, 0))
        
        # plt.gca().add_patch(PolygonPatch(cascaded_union(triangles), alpha=0.5))
        # plt.scatter(x_alpha, y_alpha, marker="*", s=20)
        
        plt.scatter(out_of_alpha[0], out_of_alpha[1], marker="*", s=20, c='blue')
        
        xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500))
        interp_cubic_geom = trim.CubicTriInterpolator(valid_triangulation, z, kind='geom')
        zi_cubic_geom = interp_cubic_geom(xi, yi)
        cs = ax.contourf(xi, yi, zi_cubic_geom, 200, cmap="Greys", vmin=0, vmax=80)
        fig.colorbar(cs, ax=ax)
        
        # plt.tricontourf(valid_triangulation, z, 50, cmap="Greys", vmin=0, vmax=90)
        
        plt.scatter(x, y, marker="o", s=10, c='red')
        plt.vlines(x_min, y_min, y_max)
        plt.vlines(x_max, y_min, y_max)
        plt.hlines(y_min, x_min, x_max)
        plt.hlines(y_max, x_min, x_max)
        plt.show()
    
    # def build_image(self):
    #
    #     x = np.array(self.velocities)
    #     y = np.array(self.distances)
    #
    #     points = np.column_stack((x, y))
    #
    #     # Define alpha parameter
    #     alpha = 0.002
    #
    #     # Generate the alpha shape
    #     alpha_shape = alphashape.alphashape(points, alpha)
    #
    #     # Initialize plot
    #     fig, ax = plt.subplots()
    #
    #     # Plot input points
    #     ax.scatter(*zip(*points))
    #
    #     # Plot alpha shape
    #     ax.add_patch(PolygonPatch(alpha_shape, alpha=.2))
    #
    #     plt.show()
    
    # def build_image(self):
    #     x_test = np.array(self.velocities)
    #     y_test = np.array(self.distances)
    #     z_test = np.array(self.colors)
    #
    #     subdiv = 3  # Number of recursive subdivisions of the initial mesh for smooth
    #     # plots. Values >3 might result in a very high number of triangles
    #     # for the refine mesh: new triangles numbering = (4**subdiv)*ntri
    #
    #     init_mask_frac = 0.0  # Float > 0. adjusting the proportion of
    #     # (invalid) initial triangles which will be masked
    #     # out. Enter 0 for no mask.
    #
    #     min_circle_ratio = .005  # Minimum circle ratio - border triangles with circle
    #     # ratio below this will be masked if they touch a
    #     # border. Suggested value 0.01 ; Use -1 to keep
    #     # all triangles.
    #
    #     random_gen = np.random.mtrand.RandomState(seed=127260)
    #
    #     # meshing with Delaunay triangulation
    #     tri = Triangulation(x_test, y_test)
    #     ntri = tri.triangles.shape[0]
    #
    #
    #     mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
    #     tri.set_mask(mask)
    #
    #
    #     # refiner = UniformTriRefiner(tri)
    #     # tri_refi, z_test_refi = refiner.refine_field(z_test, subdiv=subdiv)
    #
    #     # for the demo: loading the 'flat' triangles for plot
    #     flat_tri = Triangulation(x_test, y_test)
    #     flat_tri.set_mask(~mask)
    #
    #     plt.figure()
    #
    #     # # 1) plot of the refined (computed) data countours:
    #     # plt.tricontour(tri_refi, z_test_refi, levels=levels, cmap=cmap,
    #     #                linewidths=[2.0, 0.5, 1.0, 0.5])
    #
    #     # 3) plot of the fine mesh on which interpolation was done:
    #     plt.triplot(tri)
    #     # # 4) plot of the initial 'coarse' mesh:
    #     # plt.triplot(tri, color='0.7')
    #
    #     # plt.tricontourf(tri_refi, z_test_refi, 50, cmap="Greys", vmin=0, vmax=z_test_refi.max())
    #
    #     # # 4) plot of the unvalidated triangles from naive Delaunay Triangulation:
    #     plt.triplot(flat_tri, color='red')
    #
    #     plt.show()
    
    """
    def build_image(self):
        x = np.array(self.velocities)
        y = np.array(self.distances)
        z = np.array(self.colors)

        layout = pgo.Layout(
            paper_bgcolor='rgba(0,0,255,0.5)',
            plot_bgcolor='rgba(0,0,255,0.5)'
        )

        radio_scatter = pgo.Figure(data=pgo.Scatter(x=x,
                                               y=y,
                                               mode='markers',
                                               marker=dict(
                                                   size=10,
                                                   color=z/z.max(),
                                                   colorscale='gray',
                                                   showscale=True,
                                                   reversescale=True
                                               )
                                               )
                                   ,layout=layout
                                   )
        z = np.array(self.colors)
        print(f"\n{z.max()}")

        radio_scatter.show()
        radio_scatter.write_html('radio_image.html', auto_open=True)
    """
    
    """def build_image(self):
        x = np.array(self.velocities)
        y = np.array(self.distances)
        z = np.array(self.colors)

        triang = tri.Triangulation(x, y)

        def apply_mask(triang, alpha):
            # Mask triangles with sidelength bigger some alpha
            triangles = triang.triangles
            # Mask off unwanted triangles.
            xtri = x[triangles] - np.roll(x[triangles], 1, axis=1)
            ytri = y[triangles] - np.roll(y[triangles], 1, axis=1)
            maxi = np.max(np.sqrt(xtri ** 2 + ytri ** 2), axis=1)
            
            triang.set_mask(maxi > alpha)

        # apply_mask(triang, alpha=20)

        xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 2000), np.linspace(y.min(), y.max(), 200))

        interp_cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
        zi_cubic_geom = interp_cubic_geom(xi, yi)

        # interp_cubic_min_E = mtri.CubicTriInterpolator(triang, z, kind='min_E')
        # zi_cubic_min_E = interp_cubic_min_E(xi, yi)

        fig, ax = plt.subplots()
        ax.set_facecolor((0, 0, 0))
        print(f"\n{z.max()}")
        cs = ax.contourf(xi, yi, zi_cubic_geom, 200, cmap="Greys", extend='min')
        fig.colorbar(cs, ax=ax)
        plt.show()
        
        #
        # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 3))
        #
        # ax1.scatter(x, y, s=3, color="white")
        # ax1.set_facecolor((0, 0, 0))
        #
        # ax2.tricontourf(triang, z, 50, cmap="Greys", vmin=0, vmax=z.max())
        # ax2.set_facecolor((0, 0, 0))
        #
        # ax1.set_title("scatter")
        # ax2.set_title("with mask")
        #
        plt.show()"""
    
    def add_ray(self, ray: Ray):
        self.rays.append(ray)
        self.distances.append(ray.distance_to_point)
        self.velocities.append(ray.velocity_sign_direction * lg.norm(ray.velocity_projection))
        self.colors.append(ray.color)
