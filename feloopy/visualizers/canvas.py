import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Canvas:
    def __init__(self, x_size=8, y_size=6, n_rows=1, n_cols=1,
                 font_name='sans-serif', font_size=12, font_weight='normal',
                 font_style='normal', plot_style='ggplot', verbose=False,
                 interactive=False):
        
        # Parameters
        self.canvas_size = (x_size, y_size)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.font_name = font_name
        self.font_size = font_size
        self.font_weight = font_weight
        self.font_style = font_style
        self.plot_style = plot_style
        self.interactive = interactive
        self.verbose = verbose

        # Verbose
        if self.verbose:
            mode = "interactive" if interactive else "static"
            print(f"Creating a {mode} plot container...")

        # Containers
        self.grid_indices = []
        self.fig = None
        self.axes = None

        # Preprocessing
        self._set_style()
        self._create_subplots_grid()

    def _set_style(self):
        """Sets the plotting style based on interactivity."""
        if not self.interactive and self.verbose:
            print(f"Setting matplotlib style to '{self.plot_style}'...")
            plt.style.use(self.plot_style)

    def _create_subplots_grid(self):
        """Creates subplots based on the specified number of rows and columns."""
        if not self.interactive:
            self.fig, self.axes = plt.subplots(self.n_rows, self.n_cols, figsize=self.canvas_size)
            if self.n_rows * self.n_cols == 1:  # Ensure axes is always a list
                self.axes = [self.axes]
        else:
            self.fig = make_subplots(rows=self.n_rows, cols=self.n_cols)

    def _create_subplot(self, grid_index, dim):
        """Creates a subplot based on the grid index and dimension."""
        row, col = self._get_subplot_position(grid_index)
        if dim == 2:
            ax = self.fig.add_subplot(self.n_rows, self.n_cols, grid_index + 1)
        else:
            ax = self.fig.add_subplot(self.n_rows, self.n_cols, grid_index + 1, projection='3d')
        return ax
    
    def _get_subplot_position(self, index):
        """Gets the subplot position given an index."""
        if isinstance(index, list):
            return index[0], index[1]
        row = index // self.n_cols
        col = index % self.n_cols
        return row, col

    def _get_ax(self, grid_index=None, dim=2):
        """Gets the appropriate axis for the specified grid index."""
        if not self.interactive:
            row, col = self._get_subplot_position(grid_index)
            ax = self._create_subplot(row * self.n_cols + col, dim)
        else:
            ax = self.fig if dim == 2 else self.fig.add_subplot(self.n_rows, self.n_cols, grid_index + 1, projection='3d')
        return ax

    def _get_dim(self, z):
        """Determines the dimension based on the presence of z."""
        return 3 if z is not None else 2

    def draw_series(self, x, y, z=None, x_label=None, y_label=None, z_label=None, title=None, grid_index=[0, 0]):
        """
        Draws a series on the specified subplot.

        Parameters:
        - x, y: Data for plotting
        - z: Optional z-axis data for 3D plots
        - x_label, y_label, z_label: Labels for axes
        - title: Title of the subplot
        - grid_index: Index of the subplot
        """
        dim = self._get_dim(z)
        labels = {'x_label': x_label, 'y_label': y_label, 'z_label': z_label, 'title': title}
        
        if self.verbose:
            print(f"Drawing a series at location {grid_index}...")

        ax = self._get_ax(grid_index, dim)

        if self.verbose:
            print(f"Current ax is {ax}")

        # Plotting
        if dim == 2:
            ax.plot(x, y)
        else:
            ax.plot_trisurf(x, y, z)  # Example for 3D plotting, modify as needed

        # Set labels and title if provided
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if z_label and dim == 3:
            ax.set_zlabel(z_label)
        if title:
            ax.set_title(title)

    def show(self, legend=False, legend_loc='upper right', subplot=None, show=True, x_limit=None, y_limit=None):
        """Displays the plot with optional legend and limits."""
        if not self.interactive:
            if show:
                if legend:
                    plt.legend(loc=legend_loc)
                if x_limit:
                    plt.xlim(x_limit)
                if y_limit:
                    plt.ylim(y_limit)
                plt.show()
        else:
            self.fig.show()

canvas = Canvas