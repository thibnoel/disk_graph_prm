import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from scipy.spatial import KDTree


# Reads an image file to a numpy array
def imageToArray(img_path, transpose=False):
    """Reads an image, converts it to grayscale and stores it in a numpy array

    Args:
        img_path: (string) path to the image to read
        transpose: (bool) True if the image should be transposed

    Returns:
        img_array: the numpy array containing the image data
        img_dim: the dimensions of the returned array
    """
    img = cv2.imread(img_path, 1)
    img_array = 1 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    if transpose:
        img_array = img_array.T
    img_dim = [img_array.shape[0], img_array.shape[1]]
    return img_array, img_dim


def displayEnv(img_array):
    """Displays a numpy array as grayscale using imshow

    Args:
        img_array: the array to be displayed
    """
    plt.imshow(img_array.T, cmap=plt.cm.binary)


class OccupancyEnv:
    """2D occupancy map
    Represents a 2D occupancy grid map and provides helper functions for obstacles characterization and distance computations

    Attributes:
        map: 2D numpy array representing the occupancy
        map_dim: numpy array dimensions
        obstacles_tree: scipy KDTree holding the obstacles positions
    """
    def __init__(self):
        """Initialize attributes to None"""
        self.map = None
        self.map_dim = None
        self.obstacles_tree = None

    def setFromImg(self, img_path):
        """Initializes occupancy array from a grayscale image path"""
        self.map, self.map_dim = imageToArray(img_path)
        self.resetKDTree()

    def getObstacles(self):
        """Get the obstacles positions as a (N,2) numpy array"""
        obst = np.where(self.map == 1)
        obst = np.array(obst).T
        return obst

    def setKDTree(self):
        """Set the scipy KDTree from the obstacles positions"""
        if self.obstacles_tree is None:
            self.obstacles_tree = KDTree(self.getObstacles())

    def resetKDTree(self):
        """Reset the scipy KDTree from the obstacles positions (when map changed for ex.)"""
        self.obstacles_tree = KDTree(self.getObstacles())

    def getKDTree(self):
        """Get the current obstacles scipy KDTree"""
        self.setKDTree()
        return self.obstacles_tree

    def queryDistAt(self, pos):
        """Query the distance between a given pos. and the closest obstacle"""
        return self.obstacles_tree.query(pos)[0]

    def queryDistWitnessPointAt(self, pos):
        """Query the distance witness point between a given pos. and the closest obstacle"""
        return self.obstacles_tree.data[self.obstacles_tree.query(pos)[1]]

    def checkSegDist(self, src, dst, seg_res=0.1):
        """
        Check collision

        Args:
            src (np.ndarray): Source node
            dst (np.ndarray): Destination node

        Returns:
            bool: True if no collisions were found and False otherwise
        """
        MIN_DIST = 1  # account for pixel size, but to change if using meters
        obst_kd_tree = self.getKDTree()
        pr = seg_res
        if (dst is None) | np.all(src == dst):
            #self.dist_queries += 1
            return obst_kd_tree.query(src)[0] > MIN_DIST

        dp = dst - src
        d = np.linalg.norm(dst - src)
        steps = np.arange(0, 1., pr/d).reshape(-1, 1)
        pts = src + steps * dp
        pts = np.vstack((pts, dst))
        #self.dist_queries += 1
        return obst_kd_tree.query(pts)[0].min()

    def checkSegCollisions(self, src, dst, seg_res=0.1):
        """
        Check collision

        Args:
            src (np.ndarray): Source node
            dst (np.ndarray): Destination node

        Returns:
            bool: True if no collisions were found and False otherwise
        """
        MIN_DIST = 0  # account for pixel size, but to change if using meters
        obst_kd_tree = self.getKDTree()
        pr = seg_res
        if (dst is None) | np.all(src == dst):
            #self.dist_queries += 1
            return obst_kd_tree.query(src)[0] > MIN_DIST

        #dx, dy = dst[0] - src[0], dst[1] - src[1]
        dp = dst - src
        #yaw = np.arctan2(dy, dx)
        #d = np.hypot(dx, dy)
        d = np.linalg.norm(dst - src)
        steps = np.arange(0, 1., pr/d).reshape(-1, 1)
        pts = src + steps * dp  # np.array([np.cos(yaw), np.sin(yaw)])
        pts = np.vstack((pts, dst))
        #self.dist_queries += 1
        return bool(obst_kd_tree.query(pts)[0].min() > MIN_DIST)

    def display(self):
        displayEnv(self.map)
