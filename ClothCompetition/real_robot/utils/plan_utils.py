import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import cv2
import numpy as np
import os.path as osp

def get_target_state(env, log_dir, args):
    # get target state
    image = env.get_image()
    # image_copper = InteractiveRectangle(image)
    # plt.show(block=True)
    # img_cropped = image_copper.get_cropped_image()
    # cropped_image = np.array(img_cropped)[:,:,::-1]
    # left, upper, right, lower = image_copper.get_cropped_idx()
    cropped_image = image
    left, upper = 0, 0
    mask, approx, approx_plot = get_object_mask(cropped_image, log_dir, left, upper, image)
    # if args.env_shape == 'platform':
    #     target_pc = env.rs_listener.get_pc_given_mask(mask)
    #     mask_target = np.zeros(image.shape[:2], dtype="uint8")
    #     for point in approx[:,:]:
    #         mask_target[point[1],point[0]] = 255
    #     key_point_pose = env.rs_listener.get_pc_given_mask(mask_target)
    #
    #     # [right robot, left robot]
    #     target_picker_pos = key_point_pose[key_point_pose[:,0].argsort()][:2]
    #     if target_picker_pos[0, 1] < target_picker_pos[1, 1]:
    #         target_picker_pos[[0, 1]] = target_picker_pos[[1, 0]]
    # elif args.env_shape == 'rod':
    #     kernel = np.ones((3, 3), np.uint8)
    #     mask = cv2.erode(mask, kernel, iterations=1)
    #     target_pc = env.rs_listener.get_pc_given_mask(mask)
    #
    #     approx = approx[approx[:, 1].argsort()]
    #     if approx[0, 0] > approx[1, 0]:
    #         approx[[2, 3]] = approx[[3, 2]]
    #     else:
    #         approx[[0, 1]] = approx[[1, 0]]
    #     bias = np.array([[-3, +3], [+3, +3], [+3, -3], [-3, -3]])
    #     approx += bias
    #     key_point_pose = env.rs_listener.get_pc_given_pixel(approx)
    #     target_picker_pos = key_point_pose[:2].copy()
    # else:
    #     raise NotImplementedError
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    target_pc = env.rs_listener.get_pc_given_mask(mask)

    approx = approx[approx[:, 1].argsort()]
    if approx[0, 0] > approx[1, 0]:
        approx[[2, 3]] = approx[[3, 2]]
    else:
        approx[[0, 1]] = approx[[1, 0]]
    if args.env_shape == 'rod':
        bias = np.array([[-3, +3], [+3, +3], [+3, -3], [-3, -3]])
        approx += bias
    key_point_pose = env.rs_listener.get_pc_given_pixel(approx)
    target_picker_pos = key_point_pose[:2].copy()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(target_pc[:,0], target_pc[:,1], target_pc[:,2])
    # ax.scatter(key_point_pose[:,0], key_point_pose[:,1], key_point_pose[:,2], c='r')
    #
    # ax.set_xlabel('X Axis')
    # ax.set_ylabel('Y Axis')
    # ax.set_zlabel('Z Axis')
    # ax.set_title('3D Point Cloud')
    # ax.set_xlim(0,1)
    # ax.set_ylim(-0.5,0.5)
    # ax.set_zlim(0,1)
    # plt.show()

    target_picker_pos[:,2] = 0.0
    np.save(osp.join(log_dir, 'target_pc.npy'), target_pc)
    # target_picker_pos = approx[1:3]
    # target_picker_pos = env.rs_listener.get_pc_given_pixel(target_picker_pos)
    # get smaller two points


    return target_picker_pos, target_pc, approx, key_point_pose, approx_plot

class InteractiveRectangle:
    def __init__(self, image):

        self.original_image = Image.fromarray(image[:, :, ::-1])
        self.rect = None
        self.start_point = None
        self.end_point = None
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.original_image)
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.rid = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_click(self, event):
        # Record the start point when the left mouse button is pressed
        if event.inaxes is not None and event.button == 1:
            self.start_point = (int(event.xdata), int(event.ydata))
            # Create a rectangle which we will update with the correct size in on_release
            self.rect = Rectangle(self.start_point, 0, 0, linewidth=1, edgecolor='r', facecolor='none')
            self.ax.add_patch(self.rect)

    def on_release(self, event):
        # Record the end point when the left mouse button is released
        if event.inaxes is not None and event.button == 1:
            self.end_point = (int(event.xdata), int(event.ydata))
            self.update_rectangle()

            # Disconnect the events after the rectangle has been drawn
            self.ax.figure.canvas.mpl_disconnect(self.cid)
            self.ax.figure.canvas.mpl_disconnect(self.rid)

            # Crop and return the image
            self.crop_image()

    def update_rectangle(self):
        # Update the size of the rectangle based on the end point
        self.rect.set_width(self.end_point[0] - self.start_point[0])
        self.rect.set_height(self.end_point[1] - self.start_point[1])
        self.ax.figure.canvas.draw()

    def crop_image(self):
        # Use the start and end points to define the crop rectangle
        self.left, self.upper = self.start_point
        self.right, self.lower = self.end_point
        self.cropped_image = self.original_image.crop((self.left, self.upper, self.right,  self.lower))
        plt.close(self.fig)  # Close the interactive window
        # self.cropped_image.show()  # Show the cropped image in a new window

    def get_cropped_image(self):
        return self.cropped_image
    def get_cropped_idx(self):
        return self.left, self.upper, self.right, self.lower

def get_object_mask(cropped_image, log_dir, left, upper, image):

    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # for blue
    cloth_lower = np.array([80, 80, 40], dtype="uint8")
    cloth_upper = np.array([140, 140, 160], dtype="uint8")

    orange_lower = np.array([0, 100, 100], dtype="uint8")
    orange_upper = np.array([40, 255, 255], dtype="uint8")

    #for white
    # cloth_lower = np.array([0, 0, 100], dtype="uint8")
    # cloth_upper = np.array([255, 30, 255], dtype="uint8")

    mask = cv2.inRange(hsv, cloth_lower, cloth_upper)
    cv2.imwrite(osp.join(log_dir, 'mask_filter.png'), mask)

    # Threshold the mask to ensure it's binary
    _, thresh = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)

    ratio = 0.01
    epsilon = ratio * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    while len(approx) != 4:
        ratio +=0.005
        epsilon = ratio * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if ratio >0.1:
            raise ValueError('The object is not rectangle')
    cropped_image = cv2.drawContours(cropped_image, [approx], -1, (0, 255, 0), 2)

    # for i in range(10):
    #     _cropped_image = cropped_image.copy()
    #     epsilon = 0.01*i * cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, epsilon, True)
    #     _cropped_image = cv2.drawContours(_cropped_image, [approx], -1, (0, 255, 0), 2)
    #     cv2.imwrite(osp.join(log_dir, 'cropped_image_target_{}.png'.format(i)), _cropped_image)

    cv2.imwrite(osp.join(log_dir, 'h_target.png'), h)
    cv2.imwrite(osp.join(log_dir, 's_target.png'), s)
    cv2.imwrite(osp.join(log_dir, 'v_target.png'), v)
    cv2.imwrite(osp.join(log_dir, 'mask_target.png'), mask)
    cv2.imwrite(osp.join(log_dir, 'cropped_image_target.png'), cropped_image)

    cropped_bias = [left, upper]
    approx = approx[:,0,:] + np.array(cropped_bias)
    mask_rect = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask_rect, [approx], -1, color=255, thickness=-1)
    image = cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    # cv2.circle(image, (approx[0][0], approx[0][1]), 5, (255, 255, 255), -1)

    # cv2.imwrite(osp.join(log_dir, 'mask_o_target.png'), mask_o)
    cv2.imwrite(osp.join(log_dir, 'image_target.png'), image)

    # filter pixels in both mask and mask_o
    mask = cv2.bitwise_and(mask, mask_rect)
    cv2.imwrite(osp.join(log_dir, 'mask_target.png'), mask)
    approx_plot= approx.copy()
    return mask, approx[:, :], approx_plot