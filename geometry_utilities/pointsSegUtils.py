import numpy as np 
import matplotlib.pyplot as plt 

'''
# get a set of points aligned on a grid, based on a grid size and the (x,y) spacings between them
def get2DGridPoints(gridSize, spacing):
    points = []
    for i in range(int(gridSize[0]/spacing[0])+1):
        for j in range(int(gridSize[1]/spacing[1])+1):
            new_point = np.array([i*spacing[0],j*spacing[1]])
            points.append(new_point)
    return points

def get2DPolarPoints(max_radius, radial_res, angular_res):
    points = []
    for i in range(int(2*np.pi/angular_res) + 1):
        n_rad = int(max_radius/radial_res) + 1
        for j in range(n_rad):
            new_point = np.array([(1.*j/(n_rad-1))*max_radius*np.cos(i*angular_res), (1.*j/(n_rad - 1))*max_radius*np.sin(i*angular_res)])
            points.append(new_point)
    return points

# get a set of n uniformly-sampled 2D points in a rectangle specified by (x,y) ranges
def get2DRectUniformPoints(x_range, y_range, n_samples):
    points = []
    for k in range(n_samples):
        x = x_range[0] + np.random.random()*(x_range[1] - x_range[0])
        y = y_range[0] + np.random.random()*(y_range[1] - y_range[0])
        points.append(np.array([x,y]))
    return points


def get2DDiscUniformPoints(center_pos, radius, n_samples):
    points = []
    for k in range(n_samples):
        theta = 2*np.pi*np.random.random()
        r = radius*np.sqrt(np.random.random())
        new_p = [center_pos[0] + r*np.cos(theta), center_pos[1] + r*np.sin(theta)]
        points.append(np.array(new_p))
    return points

def get2DRingUniformPoints(center_pos, inner_rad, outer_rad, n_samples):
    points = []
    for k in range(n_samples):
        theta = 2*np.pi*np.random.random()
        r = inner_rad + (outer_rad - inner_rad)*np.sqrt(np.random.random())
        new_p = [center_pos[0] + r*np.cos(theta), center_pos[1] + r*np.sin(theta)]
        points.append(np.array(new_p))
    return points
'''

def distancesPointToSet(point, points_set):
    """Get the distances between a point and each point in a given set
    
    Args:
        point: array of size 2, the reference point to compute distances to
        points_set: array of size (N,2), the set of points to compute the distances to from the reference point
    """
    if len(points_set) == 0:
        return float('inf')
    d = (point - np.array(points_set).astype(float))
    d = np.linalg.norm(d, axis=1)
    return d
'''
def distancesPointToSegmentsSet(point, segments_set):
    if len(segments_set) == 0:
        return float('inf')
    segments_set = np.array(segments_set)
    wp_set = [pointSegDistWp(point, s) for s in segments_set]
    return distancesPointToSet(point, wp_set)

def distancesSegmentToPointSet(segment, points_set):
    if len(points_set) == 0:
        return []
    wp_list = [pointSegDistWp(p, segment) for p in points_set]
    diffs = np.array(wp_list) - np.array(points_set)
    distances = np.linalg.norm(diffs, axis=1)
    return distances

def distancesSegmentToSegmentsSet(segment, segments_set):
    segment = np.array(segment)
    segments_set = np.array(segments_set)
    self_wp_list = []
    others_wp_list = []
    for s in segments_set :
        wp_pair = segSegDistWp(segment, s)
        self_wp_list.append(wp_pair[0])
        others_wp_list.append(wp_pair[1])
    diffs = np.array(self_wp_list) - np.array(others_wp_list)
    distances = np.linalg.norm(diffs, axis=1)
    return distances

def computeAngleDiff2D(refDir, otherDir):
    normProd = np.linalg.norm(refDir)*np.linalg.norm(otherDir)
    if normProd > 1e-6 :
        angle = np.arccos(np.dot(refDir,otherDir)/normProd)
    else:
        angle = 0
    if(np.cross(refDir, otherDir) < 0):
        angle = -angle
    return angle

### Geometric distance computations in 2D
# Returns the closest point on a segment wrt a given pos
def pointSegDistWp(point, segment, toLine=False):
    dp = segment[1] - segment[0]
    seg_sqr_len = np.matmul(dp.T, dp) 
    s = (dp[0]/seg_sqr_len)*(point[0] - segment[0][0]) + (dp[1]/seg_sqr_len)*(point[1] - segment[0][1])
    
    if not toLine :
        if s < 0:
            wp = segment[0]
        elif s > 1:
            wp = segment[1]
        else:
            wp = segment[0] + s*dp 
    else:
        wp = segment[0] + s*dp 
    
    return wp 


# Segments described as 
# p1 = seg1[0] + s*u
# p2 = seg2[0] + t*v
def segSegDistWp(seg1, seg2):
    u = seg1[1] - seg1[0]
    v = seg2[1] - seg2[0]
    w = seg2[0] - seg1[0]

    uTu = np.matmul(u.T, u)
    vTv = np.matmul(v.T, v)
    wTu = np.matmul(w.T, u) 
    wTv = np.matmul(w.T, v) 
    uTv = np.matmul(u.T, v) 

    det = uTu*vTv - uTv*uTv

    # parallel case 
    if segmentsColinearityCheck(seg1, seg2) :
        candPair0 = [seg1[0], pointSegDistWp(seg1[0], seg2)]
        candPair1 = [seg1[1], pointSegDistWp(seg1[1], seg2)]
        candPair2 = [seg2[0], pointSegDistWp(seg2[0], seg1)]
        candPair3 = [seg2[1], pointSegDistWp(seg2[1], seg1)]

        candPairs = [candPair0, candPair1, candPair2, candPair3]
        candPairDists = [np.linalg.norm(cp[1] - cp[0]) for cp in candPairs]
        
        return candPairs[np.argmin(candPairDists)]

    # Compute optimal values of t and s
    s = (wTu*vTv - wTv*uTv)/det 
    t = (wTu*uTv - wTv*uTu)/det 

    # Clamp between 0 and 1
    s = max(0,min(1,s))
    t = max(0,min(1,t))

    # Recompute optim values 
    S = (t*uTv + wTu)/uTu
    T = (s*uTv - wTv)/vTv

    # Reclamp
    S = max(0,min(1,S))
    T = max(0,min(1,T))

    return seg1[0] + S*u, seg2[0] + T*v 




def pointABCLineDist(point, abcLine):
    a,b,c = abcLine[0], abcLine[1], abcLine[2]
    return np.abs(a*point[0] + b*point[1] + c)/np.sqrt(a*a + b*b)



# Returns the intersection between 2 segments (i.e. the intersection between the carrying lines and a boolean checking if it occurs on both segments)
def segmentsIntersectionCheck(segment0, segment1):
    u = segment0[1] - segment0[0]
    v = segment1[1] - segment1[0]

    # Check colinearity
    
    #if u[0]*v[1] - v[0]*u[1] < 1e-9:
    #    print("Colinear")
    #    return False, None
    
    uTu = np.matmul(u.T, u)
    vTv = np.matmul(v.T, v) 
    uTv = np.matmul(u.T, v)

    s = np.matmul((v - u*(uTv/uTu)).T, (segment1[0] - segment0[0]))*uTu / (uTv*uTv - uTu*vTv)
    t = (np.matmul(u.T, (segment1[0] - segment0[0])) + s*uTv)/uTu

    if 0 < s < 1 and 0 < t < 1 :
        return True, segment0[0] + t*u, segment1[0] + s*v
    else :
        return False, segment0[0] + t*u, segment1[0] + s*v


def segmentsColinearityCheck(seg1, seg2, thresh=1e-9):
    u = seg1[1] - seg1[0]
    v = seg2[1] - seg2[0]

    #u = u/np.linalg.norm(u)
    #v = v/np.linalg.norm(v)
    #uTu = u.T@u
    #vTv = v.T@v 
    #uTv = u.T@v 

    #det = uTu*vTv - uTv*uTv
    
    # parallel case 
    #if det < thresh :
    #    return True
    return abs(u[0]*v[1] + v[0]*u[1]) < thresh
'''