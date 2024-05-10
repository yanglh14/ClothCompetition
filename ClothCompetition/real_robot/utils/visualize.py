import os.path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

def visual_intersection(pc_pos, target_keypoints, save_path=None):
    polygon = Polygon(target_keypoints)

    fig, ax = plt.subplots()

    # Plot the rectangle (polygon)
    x, y = polygon.exterior.xy
    ax.plot(x, y, color='blue', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

    # Plot the points - color points inside the polygon differently
    for point in pc_pos:
        p = Point(point)
        if polygon.contains(p):
            ax.plot(point[0], point[1], 'o', color='green', zorder=1)
        else:
            ax.plot(point[0], point[1], 'o', color='red', zorder=1)

    # set x,y lim
    ax.set_xlim(-0.2,0.6)
    ax.set_ylim(-0.2,0.6)
    # save the plot
    plt.savefig(os.path.join(save_path, 'polygon_and_points.png'), bbox_inches='tight')