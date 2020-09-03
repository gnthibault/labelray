# Generic stuff
import argparse
import json
import os

# Numerical stuff
import numpy as np

# Geometry
import geopandas
from shapely.geometry.multilinestring import MultiLineString as MultiLineString

# Image manipulation/cv
import skimage.io
from skimage.segmentation import active_contour

# Viualization / gui
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import PolygonSelector, LassoSelector, Slider
import matplotlib.lines

class RoiManager(object):
    def __init__(self, img, ax, export_dir=None, filepath='./areas.json',
                 extractor=lambda x,y: print()):
        self.img = img
        self.ax = ax
        self.filepath = filepath
        self.extractor = extractor
        self.lines = []
        self.x_size = 1500
        self.y_size = 1500
        self.base_scale = 1.25
        self.lineprops = dict(color='purple',
                              linestyle='-',
                              linewidth = 0.5,
                              alpha=0.7,)
        self.valid_lineprops = dict()#colors=[(127,)])#,
                              #linestyles='solid',
                              #linewidths = [0.5],
                              #alpha=0.7,)
        # Snake parameters
        self.alpha = 0        # length of shape (contraction speed)
        self.beta = 1         # smoothness (higher is smoother)
        self.w_line = 0.1       # Controls attraction to brightness
        self.w_edge = 0       # Controls attraction to edges
        self.gamma = 0.025     # time stepping parameter
        self.max_px_move = 2  # Max pixel distance to move per iteration.
        self.max_iterations = 150  # Maximum iterations to optimize snake shape.
        self.boundary_condition="fixed"  # fixed or free

        # Init routine
        self.import_lines_from_file()
        self.rs = None
        self.update_ax()

        # reset selector
        self.reset_rs()

        # Check some params
        self.check_image()

        # Show help to user
        self.print_help()

        # Initialize ui
        self.init_ui()

        # Attach callbacks to plt api signals
        fig = self.ax.get_figure()
        fig.canvas.mpl_connect('scroll_event', self.zoom)
        fig.canvas.mpl_connect('key_press_event', self)

    def init_ui(self):
        axcolor = 'lightgoldenrodyellow'

        fig = self.ax.get_figure()
        fig.subplots_adjust(left=0.05, bottom=0.25)

        bx1, bx2, sx = 0.05, 0.55, 0.4
        sy = 0.03
        label_font_size = 10
        ax_alpha = fig.add_axes([bx1, 0.1, sx, sy], facecolor=axcolor)
        self.slider_alpha = Slider(ax_alpha, 'Length', 0, 2, valinit=0,
                                   valstep=0.01)
        self.slider_alpha.label.set_size(label_font_size)
        ax_beta = fig.add_axes([bx1, 0.15, sx, sy], facecolor=axcolor)
        self.slider_beta = Slider(ax_beta, 'Smooth', 0, 2, valinit=0.05,
                                   valstep=0.01)
        self.slider_beta.label.set_size(label_font_size)
        ax_w_line = fig.add_axes([bx1, 0.2, sx, sy], facecolor=axcolor)
        self.slider_w_line = Slider(ax_w_line, 'Bright', 0, 2, valinit=0.1,
                                   valstep=0.01)
        self.slider_w_line.label.set_size(label_font_size)
        ax_w_edge = fig.add_axes([bx1, 0.25, sx, sy], facecolor=axcolor)
        self.slider_edge = Slider(ax_w_edge, 'Edge', 0, 2, valinit=0,
                                   valstep=0.01)
        self.slider_edge.label.set_size(label_font_size)
        ax_gamma = fig.add_axes([bx2, 0.15, sx, sy], facecolor=axcolor)
        self.slider_gamma = Slider(ax_gamma, 'Step', 0.001, 1, valinit=0.005,
                                   valstep=0.001)
        self.slider_gamma.label.set_size(label_font_size)
        ax_max_px_move = fig.add_axes([bx2, 0.20, sx, sy], facecolor=axcolor)
        self.slider_max_px_move = Slider(ax_max_px_move, 'Move px', 0, 5, valinit=2,
                                   valstep=1)
        self.slider_max_px_move.label.set_size(label_font_size)
        ax_max_iterations = fig.add_axes([bx2, 0.25, sx, sy], facecolor=axcolor)
        self.slider_max_iterations = Slider(ax_max_iterations, 'Nb iter', 0, 200, valinit=10,
                                   valstep=1)
        self.slider_max_iterations.label.set_size(label_font_size)

        def update(val):
            self.alpha = self.slider_alpha.val
            self.beta = self.slider_beta.val
            self.w_line = self.slider_w_line.val
            self.edge = self.slider_edge.val
            self.gamma = self.slider_gamma.val
            self.max_px_move = self.slider_max_px_move.val
            self.max_iterations = self.slider_max_iterations.val
            self.smooth_current_polygon()
            self.redraw()

        self.slider_alpha.on_changed(update)
        self.slider_beta.on_changed(update)
        self.slider_w_line.on_changed(update)
        self.slider_edge.on_changed(update)
        self.slider_gamma.on_changed(update)
        self.slider_max_px_move.on_changed(update)
        self.slider_max_iterations.on_changed(update)


    def reset_rs(self):
        self.rs = PolygonSelector(self.ax,
                                  onselect=self.select_polygon,
                                  useblit=True,
                                  lineprops=self.lineprops,
                                  markerprops=None,
                                  vertex_select_radius=15)
#        self.rs = LassoSelector(self.ax,
#                                  onselect=self.select_polygon,
#                                  useblit=True,
#                                  lineprops=self.lineprops)

        self.rs.set_active(False)


    def redraw(self):
        fig = self.ax.get_figure()
        fig.canvas.draw()

    def zoom(self, event):
        # get the current x and y limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/self.base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = self.base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(f"Unhandled event button pressed: {event.button}")
        # set new limits
        self.ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        self.ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        # Now redraw
        self.redraw()

    def print_help(self):
        print("-Press key N or n to start selecting new line.\n"
              "-Print key V or v to validate current selection\n"
              "-Print key A or a to abort current selections\n"
              "-Print key E or e to extract all selections\n"
              "-Press key Q or q to quit\n")

    def check_image(self):
        pass

    def import_lines_from_file(self):
        if os.path.exists(self.filepath):
            print(f"Reading file {self.filepath}")
            mls = geopandas.GeoDataFrame.from_file(self.filepath).geometry.values
            self.lines = [np.array([c for c in l.coords]) for l in mls]

    def export_lines_to_file(self):
        export_dir = os.path.dirname(self.filepath)
        if not os.path.exists(export_dir):
            print(f"Path {export_dir} does not exists, creating it...")
        mls = MultiLineString(self.lines)
        gpd = geopandas.GeoDataFrame({"geometry": mls})

        with open(self.filepath, 'w') as f:
            print(f"Exporting to file {self.filepath}")
            f.write(gpd.to_json())

    def update_ax(self):
        _ = [p.remove() for p in reversed(self.ax.patches)]
        if self.rs is not None:
            self.rs.set_active(False)
        # Create patch collection with specified colour/alpha
        pc = LineCollection(self.lines, *self.valid_lineprops)
        # Add collection to axes
        self.ax.add_collection(pc)

    def smooth_line(self, coord_list):
        if coord_list.shape[0] < 5:
            pass
        snake = active_contour(self.img,
                               coord_list,
                               alpha=self.alpha,
                               beta=self.beta,
                               w_line=self.w_line,
                               w_edge=self.w_edge,
                               gamma=self.gamma,
                               max_px_move=self.max_px_move,
                               max_iterations=self.max_iterations,
                               boundary_condition=self.boundary_condition)
        return ([i[0] for i in snake], #+snake[0][0],
                [i[1] for i in snake])#+snake[0][1])

    def __call__(self, event):
        """ The callback that will filter and use specific UI events.
        """
        if event.key in ['A', 'a'] and self.rs.active:
            print('RectangleSelector deactivated.')
            self.reset_rs()
            self.redraw() # force re-draw
        if event.key in ['N', 'n'] and not self.rs.active:
            print('RectangleSelector activated.')
            self.reset_rs()
            self.rs.set_active(True)
        if event.key in ['V', 'v'] and self.rs.active:
            print('Adding current line selection to list.')
            self.lines.append(np.array([self.rs._xs[:-1],self.rs._ys[:-1]]).T)
            self.reset_rs()
            self.update_ax()
            self.redraw() # force re-draw
        if event.key in ['E', 'e']:
            print('Now extracting data')
            self.export_lines_to_file()

    def select_polygon(self, eclick):
        """
            eclick is the list of coordinates as a list of tuple [(x,y),...]
        """
        #print(self.rs.line)
        #self.rs.line.set_picker(True)
        #self.rs2 = mpl.lines.VertexSelector(self.rs.line)
        self.smooth_current_polygon()

    def smooth_current_polygon(self):
        """
            uses spline and image to smooth
        """
        if self.rs is None:
            return
        #Now smooth out in order to have a nice line
        coords = np.array(list(zip(self.rs._xs, self.rs._ys)))
        if np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        xs, ys = self.smooth_line(coords)
        self.rs._xs, self.rs._ys = xs+[xs[0]], ys+[ys[0]]
        self.rs._draw_polygon()


def main(input_path, output_path):
    # Load cube
    img = skimage.io.imread(input_path)

    # plot stuff
    fig, ax = plt.subplots(1, figsize=(16,12))
    ax.imshow(img)

    # do some stuff
    rman = RoiManager(img, ax, filepath=output_path)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Use it wisely')
    parser.add_argument('--input_path', '-i', dest='input_path', type=str,
                        required=True, help='Path to the data directory')
    parser.add_argument('--output_path', '-o', dest='output_path', type=str,
                        required=True, help='output extraction path')
    args = parser.parse_args()
    main(input_path=args.input_path,
         output_path=args.output_path)
