import numpy as np
import pandas as pd
import math
import scipy.ndimage as nd
import cv2
import skimage as ski
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np

class BaseClass:
    def __init__(self,filename,edge_length = 208,curvature=0.07179,rmax=18.3,rmin=14.1,resolution=0.2,thickness=6.78):
        self.edge_length = edge_length
        self.curvature = curvature
        self.rmax = rmax
        self.rmin = rmin
        #self.rstar = (rmax + rmin) / 2
        # suppose the resolution of the TEM is 0.2 nm
        self.resolution = resolution

        # these values should be filled
        self.pr = None
        self.pc = None
        self.prism_surf = None
        self.face_proj = None
        self.tag_prob = None

        self.Rmax = int(rmax / resolution)
        self.Rmin = int(rmin / resolution)
        self.dthick = int(thickness / resolution)

        self.Rc = int(curvature * edge_length / resolution)
        self.Pedge = int(edge_length / resolution)
        self.Pedge_h = int(edge_length / resolution * math.sqrt(3) / 2)
        self.o1r, self.o1c = self.Pedge_h - 1 - self.Rc, int(math.sqrt(3) * self.Rc)
        self.df = pd.read_excel(filename)
        self.create_masks()
        self.bind_prob = np.zeros_like(self.prism_surf)

    def dist2o1(self, x1, y1, ):
        dis = math.sqrt((x1 - self.o1c) ** 2 + (y1 - self.o1r) ** 2)
        return dis

    def tag_kernel(self):
        Rmax = self.Rmax
        Rmin = self.Rmin
        tag_space = np.ones((1 + 2 * Rmax, 1 + 2 * Rmax, 1 + 2 * Rmax))
        tag_space[Rmax, Rmax, Rmax] = 0
        distance_mask = nd.morphology.distance_transform_edt(tag_space)
        # distance_mask = np.round(distance_mask)
        tag_prob = np.zeros_like(distance_mask)
        bind = (distance_mask <= Rmax) & (distance_mask >= Rmin)
        tag_prob[bind] = 1
        below = distance_mask < Rmin
        tag_prob[below] = -1389284  # total volume in the Rmin and Rmax ring

        return tag_prob


    def face_prob(self):
        Rmax= self.Rmax
        Rmin= self.Rmin
        apr_r_Rmin = int(np.round(math.sqrt(self.Rmax ** 2 - self.Rmin ** 2)))
        mask = np.ones((Rmax * 2 + 1, Rmax * 2 + 1, Rmax * 2 + 1), dtype=bool)
        mask[Rmax, Rmax, Rmax] = 0
        distance_mask = nd.morphology.distance_transform_edt(mask)
        over = distance_mask > Rmax
        small = distance_mask < Rmin
        filter_proj = np.copy(distance_mask)
        filter_proj[over] = 0
        filter_proj[~over] = 1
        filter_proj[small] = 0
        filter_proj[:, :, :Rmax + Rmin + 1] = 0
        face_proj = np.sum(filter_proj, axis=2)
        face_proj = face_proj / np.sum(face_proj)
        face_proj = face_proj[Rmax-apr_r_Rmin:Rmax+apr_r_Rmin+1,Rmax-apr_r_Rmin:Rmax+apr_r_Rmin+1]
        return face_proj

    def prismsurface(self):
        ## First get a polygon of the lines
        ## get points of circle
        ## put the circle and the line pt together and mark them use img[rr,cc]= 1
        Pedge_h = self.Pedge_h
        Pedge = self.Pedge
        dthick=self.dthick
        Rc = self.Rc
        prism3D = np.zeros((Pedge_h, Pedge, dthick))

        [b_y, a_y, e_y, f_y, c_y, d_y] = [Pedge_h - 1, Pedge_h - int(1.5 * Rc) - 1, int(1.5 * Rc), int(1.5 * Rc),
                                          Pedge_h - int(1.5 * Rc) - 1, Pedge_h - 1]
        [b_x, a_x, e_x, f_x, c_x, d_x] = [int(math.sqrt(3) * Rc), int(math.sqrt(3) * Rc / 2),
                                          Pedge // 2 - int(math.sqrt(3) * Rc / 2),
                                          Pedge // 2 + int(math.sqrt(3) * Rc / 2),
                                          Pedge - int(math.sqrt(3) * Rc / 2) - 1, Pedge - int(math.sqrt(3) * Rc) - 1]

        l1r, l1c = ski.draw.line(b_y, b_x, d_y, d_x)
        l2r, l2c = ski.draw.line(a_y, a_x, e_y, e_x)
        l3r, l3c = ski.draw.line(f_y, f_x, c_y, c_x)
        pr = np.concatenate((l1r, l2r, l3r))
        pc = np.concatenate((l1c, l2c, l3c))

        # center of the tip circles
        o1r, o1c = Pedge_h - 1 - Rc, int(math.sqrt(3) * Rc)
        o2r, o2c = Pedge_h - 1 - Rc, Pedge - 1 - int(math.sqrt(3) * Rc)
        o3r, o3c = 2 * Rc, Pedge // 2
        # create the circular tip of prism
        c1r, c1c = ski.draw.circle_perimeter(o1r, o1c, Rc)

        # remove the portion of circle that does not belong to the tip of prism
        # lab fits the equation y = math.sqrt(3)*(x-math.sqrt(3)*Rc)+Pedge_h-1
        # if c1r < calculated r, it is inside the prism not tip region
        cal1r = math.sqrt(3) * (c1c - math.sqrt(3) * Rc) + Pedge_h - 1
        yes = cal1r <= c1r
        tip1r = c1r[yes]
        tip1c = c1c[yes]
        # tip 2 is a mirror image of tip 1, so they share same y, opposite x
        tip2c = Pedge - 1 - tip1c
        # tip 3, all y above 1.5r is not on the tip

        c3r, c3c = ski.draw.circle_perimeter(o3r, o3c, Rc)
        yes = c3r <= int(1.5 * Rc)
        tip3r = c3r[yes]
        tip3c = c3c[yes]

        pr = np.concatenate((pr, tip1r, tip1r, tip3r))
        pc = np.concatenate((pc, tip1c, tip2c, tip3c))
        prism3D[pr, pc, :] = 1

        polygon = np.array([[b_y, b_x], [d_y, d_x], [c_y, c_x], [f_y, f_x], [e_y, e_x], [a_y, a_x]])
        face_mask = ski.draw.polygon2mask((prism3D.shape[0], prism3D.shape[1]), polygon)
        face_mask = face_mask.astype(float)
        face_mask = cv2.circle(face_mask, (o1c, o1r), Rc, (1, 0, 0), -1)
        face_mask = cv2.circle(face_mask, (o2c, o2r), Rc, (1, 0, 0), -1)
        face_mask = cv2.circle(face_mask, (o3c, o3r), Rc, (1, 0, 0), -1)
        prism3D[:, :, 0] = face_mask
        prism3D[:, :, -1] = face_mask

        self.prism_surf = prism3D
        self.pr = pr
        self.pc= pc

    def tip_idx(self):
        # center of the tip circles
        o1r, o1c = self.Pedge_h - 1 - self.Rc, int(math.sqrt(3) * self.Rc)

        # create the circular tip of prism
        c1r, c1c = ski.draw.circle_perimeter(o1r, o1c, self.Rc)

        # remove the portion of circle that does not belong to the tip of prism
        # lab fits the equation y = math.sqrt(3)*(x-math.sqrt(3)*Rc)+Pedge_h-1
        # if c1r < calculated r, it is inside the prism not tip region
        cal1r = math.sqrt(3) * (c1c - math.sqrt(3) * self.Rc) + self.Pedge_h - 1
        yes = cal1r <= c1r
        tip1r = c1r[yes]
        tip1c = c1c[yes]

        return tip1r, tip1c

    def create_masks(self):
        self.face_proj = self.face_prob()
        self.tag_prob = self.tag_kernel()
        self.prismsurface()

    def show_prismface(self):
        plt.imshow(self.prism_surf[:,:,0])

    def show_face_proj(self):
        plt.imshow(self.face_proj)
    def show_bindface(self):
        plt.imshow(self.bind_prob[:,:,0])
    def show_bindtip(self):
        tr, tc = self.tip_idx()
        plt.imshow(self.bind_prob[tr,tc])

    def save_bindmap(self,filepath):
        self.bind_prob[:,:,-1]=self.bind_prob[:,:,0]
        np.save(filepath,self.bind_prob)
    def visualize3D(self):
        self.bind_prob[:, :, -1] = self.bind_prob[:, :, 0]
        sur = self.prism_surf == 1
        [x, y, z] = np.meshgrid(np.linspace(1, 1040, 1040), np.linspace(1, 900, 900), np.linspace(1, 33, 33))
        X = x[sur]
        Y = y[sur]
        Z = z[sur]
        dat = self.bind_prob[sur]
        points = np.array([X, Y, Z, dat]).T

        point_cloud = pv.PolyData(points[:, :-1])
        point_cloud["elevation"] = points[:, -1]
        point_cloud.plot(render_points_as_spheres=True)

if __name__ == "__main__":
    cls = BaseClass('Probability density map/automated position classify.xlsx')
    #print(cls.face_proj.shape)
    face_prob = cls.face_prob()
    plt.imshow(face_prob)
    plt.colorbar()
    plt.savefig('face_kernel.eps')


    #cls.visualize3D()