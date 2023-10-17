import numpy as np
import pandas as pd
import math
import scipy.ndimage as nd
import cv2
import skimage as ski
import matplotlib.pyplot as plt
from base_model import BaseClass
import pyvista as pv

class heatmap(BaseClass):
    def __init__(self,filename):
        super().__init__(filename)


    def add_face(self,prism_prob, px, py, point):

        Pedge = self.Pedge
        apr_r_Rmin = int(np.round(math.sqrt(self.Rmax ** 2 - self.Rmin ** 2)))
        if py > self.Pedge_h:
            print("point ", point, "not on the face")
        else:
            print(self.prism_surf.shape)
            if (px - apr_r_Rmin >= 0) & (px + apr_r_Rmin < Pedge - 1) & (py + apr_r_Rmin < self.Pedge_h - 1):
                pad = self.prism_surf[py - apr_r_Rmin:py + apr_r_Rmin + 1, px - apr_r_Rmin:px + apr_r_Rmin + 1, 0]
                conv = self.face_proj * pad
                conv_sum = conv.sum()
                if conv_sum > 0:
                    prism_prob[py - apr_r_Rmin:py + apr_r_Rmin + 1, px - apr_r_Rmin:px + apr_r_Rmin + 1,
                    0] += conv / conv_sum
            elif (px - apr_r_Rmin < 0):
                print("x+ apr_r_Rmin<0")
                right_x = px + apr_r_Rmin + 1
                top_y = min(self.Pedge_h, py + apr_r_Rmin + 1) - (py - apr_r_Rmin)
                new_face = self.face_proj[:top_y, -right_x:]
                pad = self.prism_surf[py - apr_r_Rmin:top_y + py - apr_r_Rmin, :right_x, 0]
                conv = new_face * pad
                conv_sum = conv.sum()
                if conv_sum > 0:
                    prism_prob[py - apr_r_Rmin:py - apr_r_Rmin + top_y, :right_x, 0] += conv / conv_sum


            elif (px + apr_r_Rmin > Pedge - 1):
                print("x+ apr_r_Rmin<Pedge-1")
                right_x = Pedge - (px - apr_r_Rmin)
                top_y = min(self.Pedge_h, py + apr_r_Rmin + 1) - (py - apr_r_Rmin)
                new_face = self.face_proj[:top_y, :right_x]
                pad = self.prism_surf[py - apr_r_Rmin: top_y + (py - apr_r_Rmin), :, 0]
                conv = new_face * pad
                conv_sum = conv.sum()
                if conv_sum > 0:
                    prism_prob[py - apr_r_Rmin:py - apr_r_Rmin + top_y, -right_x:, 0] += conv / conv_sum

        return prism_prob

    def add_faceEvents(self):
        prism_prob = np.zeros_like(self.prism_surf)
        df_face = self.df[self.df['face'] == 1]
        yarray = df_face['convert_y'].to_numpy()
        xarray = df_face['convert_x'].to_numpy()

        for point in range(xarray.size):
            print("processing point", point)
            px = int(xarray[point] * self.edge_length / self.resolution)
            py = self.Pedge_h - int(yarray[point] * self.edge_length / self.resolution)
            prism_prob = self.add_face(prism_prob, px, py, point)

        self.bind_prob +=prism_prob

    def add_tipEvents(self,verbose=False,zverbose=False):
        df_tip = self.df[self.df['tip']==1]
        yarray = df_tip['convert_y'].to_numpy()
        xarray = df_tip['convert_x'].to_numpy()

        for point in range(xarray.size):
            x = xarray[point]
            y = yarray[point]

            if (-y < 0.08557):
                dis2edge = -int(y * self.edge_length / self.resolution)
                Xx = int(x * self.edge_length / self.resolution)
                prism_bindmap = self.add_tip(dis2edge, Xx, point, zverbose=zverbose)
                self.bind_prob += prism_bindmap

                if verbose:
                    print(x, y, "point", point, "sum of binding", prism_bindmap.sum())

            else:
                print("point", point, x, y, "R* > Rmax")




    def add_tip(self,dis2edge,Xx,point,zverbose=False):
        bind_prob = np.zeros_like(self.prism_surf)
        # Rap here is the distance of the tag to the particle
        # dis2edge<0 if inside prims, dis2edge if on the tip
        dis2tipc = self.dist2o1(Xx, self.Pedge_h - 1 + dis2edge)
        Pedge_h = self.Pedge_h
        Pedge = self.Pedge
        Rmax = self.Rmax
        dthick = self.dthick
        Rap = dis2tipc - self.Rc
        tag_prob = self.tag_prob
        if Rap > self.Rmax:
            print("Tip processor point", point, "not binding")

        else:
            hmax = int(math.sqrt(self.Rmax ** 2 - Rap ** 2))
            if zverbose:
                print("point", point, "Rap", Rap)

            binding_space = np.zeros((Pedge_h + 1 + Rmax + dis2edge, Pedge, dthick + 2 * hmax + 2 * Rmax + 1))
            binding_space[:Pedge_h, :Pedge, Rmax + hmax:Rmax + hmax + dthick] = self.prism_surf
            bindmap1 = np.zeros_like(binding_space)
            # solve case where only half kernel involved
            left_x = max(0, Xx - Rmax)
            right_x = min(Xx + Rmax + 1, Pedge)
            x_range = right_x - left_x
            actual_tag_prob = np.zeros((tag_prob.shape[0], right_x - left_x, tag_prob.shape[2]))
            if left_x == 0:
                actual_tag_prob[:, :, :] = tag_prob[:, -x_range:, :]
            else:
                actual_tag_prob[:, :, :] = tag_prob[:, :x_range, :]

            for i in range(Rmax, Rmax + dthick + 1):
                pad = binding_space[dis2edge + Pedge_h - 1 - Rmax:Pedge_h + dis2edge + Rmax, left_x:right_x,
                      i - Rmax:Rmax + i + 1]

                conv = actual_tag_prob * pad
                conv_sum = conv.sum()
                if zverbose:
                    conv_sum = conv.sum()
                    print("hieght", i, conv_sum)
                    print("pad sum", pad.sum())

                if conv_sum >= 0:
                    bindmap1[dis2edge + Pedge_h - 1 - Rmax:Pedge_h + dis2edge + Rmax, left_x:right_x,
                    i - Rmax:Rmax + i + 1] += conv
                else:
                    break

            flip = bindmap1[self.pr,self.pc, Rmax + hmax:Rmax + hmax + dthick]
            edge = flip + np.flip(flip, 1)
            face = bindmap1[:Pedge_h - 1, :, Rmax + hmax]
            total = edge.sum() + face.sum() * 2
            if total > 0:
                bind_prob = np.zeros_like(self.prism_surf)
                bind_prob[:-1, :, 0] = bindmap1[:Pedge_h - 1, :, Rmax + hmax] / total  # face binding map
                bind_prob[self.pr, self.pc, :] = edge[:, :] / total  # edge binding

        return bind_prob

    def add_edge(self,dist2edge,Xx, zverbose=False):
        bind_prob = np.zeros_like(self.prism_surf)
        Rmax = self.Rmax
        Pedge_h = self.Pedge_h
        Pedge = self.Pedge
        dthick = self.dthick
        tag_prob = self.tag_prob
        # dist2edge here is the distance of the tag to the particle, when y<0 dist2edge>0
        hmax = int(math.sqrt(Rmax ** 2 - dist2edge ** 2))
        binding_space = np.zeros((Pedge_h + 1 + Rmax + dist2edge, Pedge, dthick + 2 * hmax + 2 * Rmax + 1))
        binding_space[:Pedge_h, :Pedge, Rmax + hmax:Rmax + hmax + dthick] = self.prism_surf
        bindmap1 = np.zeros_like(binding_space)
        # solve case where only half kernel involved
        left_x = max(0, Xx - Rmax)
        right_x = min(Xx + Rmax + 1, Pedge)
        x_range = right_x - left_x
        actual_tag_prob = np.zeros((tag_prob.shape[0], right_x - left_x, tag_prob.shape[2]))
        if left_x == 0:
            actual_tag_prob[:, :, :] = tag_prob[:, -x_range:, :]
        else:
            actual_tag_prob[:, :, :] = tag_prob[:, :x_range, :]

        for i in range(Rmax, Rmax + dthick + 1):
            pad = binding_space[dist2edge + Pedge_h - 1 - Rmax:Pedge_h + dist2edge + Rmax, left_x:right_x,
                  i - Rmax:Rmax + i + 1]

            conv = actual_tag_prob * pad
            conv_sum = conv.sum()
            if zverbose:
                conv_sum = conv.sum()
                print("hieght", i, "conv sum", conv_sum)
                print("pad sum", pad.sum())

            if conv_sum >= 0:
                bindmap1[dist2edge + Pedge_h - 1 - Rmax:Pedge_h + dist2edge + Rmax, left_x:right_x,
                i - Rmax:Rmax + i + 1] += conv
            else:
                break

        flip = bindmap1[:Pedge_h, :, Rmax + hmax:Rmax + hmax + dthick]
        edge = flip + np.flip(flip, 2)
        face = bindmap1[:Pedge_h - 1, :, Rmax + hmax]
        total = edge.sum() + face.sum() * 2
        if total > 0:
            bind_prob = np.zeros_like(self.prism_surf)
            bind_prob[:-1, :, 0] = bindmap1[:Pedge_h - 1, :, Rmax + hmax] / total  # face binding map
            bind_prob[:, :, :] = edge[:, :, :] / total  # edge binding

        return bind_prob



    def add_edgeEvents(self,verbose=False,zverbose=False):
        df_edge = self.df[self.df['edge'] == 1]
        yarray = df_edge['convert_y'].to_numpy()
        xarray = df_edge['convert_x'].to_numpy()

        for point in range(xarray.size):
            x = xarray[point]
            y = yarray[point]
            print("point", point, x, y)
            if (-y < 0.08557):

                dis2edge = -int(y * self.edge_length / self.resolution)
                Xx = int(x * self.edge_length / self.resolution)
                prism_bindmap = self.add_edge(dis2edge, Xx,  zverbose=zverbose)
                self.bind_prob += prism_bindmap
                if verbose:
                    print(x, y, "point", point, "sum of binding", prism_bindmap.sum())

            elif -y >= 0.08557:
                print("point", point, x, y, "R* > Rmax, not binding")

    def add_allEvents(self,verbose=False,zverbose=False):
        self.add_tipEvents(verbose,zverbose)
        self.add_edgeEvents(verbose,zverbose)
        self.add_faceEvents()
        self.bind_prob[:,:,-1]=self.bind_prob[:,:,0]



    def show3D(self):
        data = self.bind_prob
        sur = self.prism_surf == 1
        [x, y, z] = np.meshgrid(np.linspace(1, 1040, 1040), np.linspace(1, 900, 900), np.linspace(1, 33, 33))
        X = x[sur]
        Y = y[sur]
        Z = z[sur]
        dat = data[sur]
        points = np.array([X, Y, Z, dat]).T

        point_cloud = pv.PolyData(points[:, :-1])
        point_cloud["elevation"] = points[:, -1]
        point_cloud.plot(render_points_as_spheres=True)




