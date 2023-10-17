from system_info import *
import skimage as ski
import math
import matplotlib.pyplot as plt
import numpy as np

x2=int(math.sqrt(3)*Rc)
y2=Pedge_h-1-Rc

def dist2o1(x1,y1,x2=x2,y2=y2):
    dis = math.sqrt((x1-x2)**2+(y1-y2)**2)
    return dis

def tag_kernel(Rmax,Rmin):
    tag_space = np.ones((1+2*Rmax,1+2*Rmax,1+2*Rmax))
    tag_space[Rmax,Rmax,Rmax]=0
    distance_mask = nd.morphology.distance_transform_edt(tag_space)
    #distance_mask = np.round(distance_mask)
    tag_prob = np.zeros_like(distance_mask)
    bind=(distance_mask<=Rmax) & (distance_mask>=Rmin)
    tag_prob[bind]=1
    below = distance_mask < Rmin
    tag_prob[below]=-1389284 # total volume in the Rmin and Rmax ring
    return tag_prob

def tag_kernel_discrete(Rmax,Rmin):
    tag_space = np.ones((1+2*Rmax,1+2*Rmax,1+2*Rmax))
    tag_space[Rmax,Rmax,Rmax]=0
    distance_mask = nd.morphology.distance_transform_edt(tag_space)
    #distance_mask = np.round(distance_mask)
    tag_prob = np.zeros_like(distance_mask)
    bind=(distance_mask<(Rmax+Rmin)//2+1) & (distance_mask>(Rmax+Rmin)//2-1)
    tag_prob[bind]=1
    below = distance_mask < Rmin
    tag_prob[below]=-80030 # total volume in the Rmin and Rmax ring
    return tag_prob
def face_prob(apr_r_Rmin):

    print("2D_R_at Rmin Z height", apr_r_Rmin)
    mask = np.ones((2*apr_r_Rmin+1,2*apr_r_Rmin+1),dtype = bool)
    mask[apr_r_Rmin,apr_r_Rmin] = 0
    distance_mask = nd.morphology.distance_transform_edt(mask)
    over = distance_mask > apr_r_Rmin
    face_prob = np.copy(distance_mask)
    face_prob[over] = 0
    face_prob[~over] = 1
    return face_prob

def prismsurface(Pedge_h,Pedge,dthick):
    ## First get a polygon of the lines
    ## get points of circle
    ## put the circle and the line pt together and mark them use img[rr,cc]= 1
    prism3D = np.zeros((Pedge_h,Pedge,dthick))


    [b_y,a_y,e_y,f_y,c_y,d_y] = [Pedge_h-1,Pedge_h-int(1.5*Rc)-1,int(1.5*Rc),int(1.5*Rc),Pedge_h-int(1.5*Rc)-1,Pedge_h-1]
    [b_x,a_x,e_x,f_x,c_x,d_x] = [int(math.sqrt(3)*Rc),int(math.sqrt(3)*Rc/2),Pedge//2-int(math.sqrt(3)*Rc/2),Pedge//2+int(math.sqrt(3)*Rc/2),Pedge-int(math.sqrt(3)*Rc/2)-1,Pedge-int(math.sqrt(3)*Rc)-1]

    l1r,l1c = ski.draw.line(b_y,b_x,d_y,d_x)
    l2r,l2c = ski.draw.line(a_y,a_x,e_y,e_x)
    l3r,l3c = ski.draw.line(f_y,f_x,c_y,c_x)
    pr = np.concatenate((l1r,l2r,l3r))
    pc = np.concatenate((l1c,l2c,l3c))

    # center of the tip circles
    o1r,o1c = Pedge_h-1-Rc,int(math.sqrt(3)*Rc)
    o2r,o2c = Pedge_h-1-Rc,Pedge-1-int(math.sqrt(3)*Rc)
    o3r,o3c = 2*Rc,Pedge//2
    # create the circular tip of prism
    c1r,c1c = ski.draw.circle_perimeter(o1r,o1c,Rc)

    # remove the portion of circle that does not belong to the tip of prism
    # lab fits the equation y = math.sqrt(3)*(x-math.sqrt(3)*Rc)+Pedge_h-1
    # if c1r < calculated r, it is inside the prism not tip region
    cal1r= math.sqrt(3)*(c1c-math.sqrt(3)*Rc)+Pedge_h-1
    yes = cal1r<=c1r
    tip1r = c1r[yes]
    tip1c = c1c[yes]
    #tip 2 is a mirror image of tip 1, so they share same y, opposite x
    tip2c = Pedge-1-tip1c
    #tip 3, all y above 1.5r is not on the tip

    c3r,c3c = ski.draw.circle_perimeter(o3r,o3c,Rc)
    yes = c3r<=int(1.5*Rc)
    tip3r = c3r[yes]
    tip3c = c3c[yes]

    pr = np.concatenate((pr,tip1r,tip1r,tip3r))
    pc = np.concatenate((pc,tip1c,tip2c,tip3c))
    prism3D[pr,pc,:]=1

    polygon = np.array([[b_y,b_x],[d_y,d_x],[c_y,c_x],[f_y,f_x],[e_y,e_x],[a_y,a_x]])
    face_mask= ski.draw.polygon2mask((prism3D.shape[0],prism3D.shape[1]),polygon)
    face_mask = face_mask.astype(float)
    face_mask=cv2.circle(face_mask,(o1c,o1r),Rc,(1,0,0),-1)
    face_mask = cv2.circle(face_mask,(o2c,o2r),Rc,(1,0,0),-1)
    face_mask = cv2.circle(face_mask,(o3c,o3r),Rc,(1,0,0),-1)
    prism3D[:,:,0]=face_mask
    prism3D[:,:,-1]=face_mask

    return prism3D

def tip_edge_contour_idx(Pedge_h,Pedge,dthick):
    ## First get a polygon of the lines
    ## get points of circle
    ## put the circle and the line pt together and mark them use img[rr,cc]= 1
    prism3D = np.zeros((Pedge_h,Pedge,dthick))


    [b_y,a_y,e_y,f_y,c_y,d_y] = [Pedge_h-1,Pedge_h-int(1.5*Rc)-1,int(1.5*Rc),int(1.5*Rc),Pedge_h-int(1.5*Rc)-1,Pedge_h-1]
    [b_x,a_x,e_x,f_x,c_x,d_x] = [int(math.sqrt(3)*Rc),int(math.sqrt(3)*Rc/2),Pedge//2-int(math.sqrt(3)*Rc/2),Pedge//2+int(math.sqrt(3)*Rc/2),Pedge-int(math.sqrt(3)*Rc/2)-1,Pedge-int(math.sqrt(3)*Rc)-1]

    l1r,l1c = ski.draw.line(b_y,b_x,d_y,d_x)
    l2r,l2c = ski.draw.line(a_y,a_x,e_y,e_x)
    l3r,l3c = ski.draw.line(f_y,f_x,c_y,c_x)
    pr = np.concatenate((l1r,l2r,l3r))
    pc = np.concatenate((l1c,l2c,l3c))

    # center of the tip circles
    o1r,o1c = Pedge_h-1-Rc,int(math.sqrt(3)*Rc)
    o2r,o2c = Pedge_h-1-Rc,Pedge-1-int(math.sqrt(3)*Rc)
    o3r,o3c = 2*Rc,Pedge//2
    # create the circular tip of prism
    c1r,c1c = ski.draw.circle_perimeter(o1r,o1c,Rc)

    # remove the portion of circle that does not belong to the tip of prism
    # lab fits the equation y = math.sqrt(3)*(x-math.sqrt(3)*Rc)+Pedge_h-1
    # if c1r < calculated r, it is inside the prism not tip region
    cal1r= math.sqrt(3)*(c1c-math.sqrt(3)*Rc)+Pedge_h-1
    yes = cal1r<=c1r
    tip1r = c1r[yes]
    tip1c = c1c[yes]
    #tip 2 is a mirror image of tip 1, so they share same y, opposite x
    tip2c = Pedge-1-tip1c
    #tip 3, all y above 1.5r is not on the tip

    c3r,c3c = ski.draw.circle_perimeter(o3r,o3c,Rc)
    yes = c3r<=int(1.5*Rc)
    tip3r = c3r[yes]
    tip3c = c3c[yes]

    pr = np.concatenate((pr,tip1r,tip1r,tip3r))
    pc = np.concatenate((pc,tip1c,tip2c,tip3c))

    return pr,pc

def tip_idx(Pedge_h):
    # center of the tip circles
    o1r,o1c = Pedge_h-1-Rc,int(math.sqrt(3)*Rc)

    # create the circular tip of prism
    c1r,c1c = ski.draw.circle_perimeter(o1r,o1c,Rc)

    # remove the portion of circle that does not belong to the tip of prism
    # lab fits the equation y = math.sqrt(3)*(x-math.sqrt(3)*Rc)+Pedge_h-1
    # if c1r < calculated r, it is inside the prism not tip region
    cal1r = math.sqrt(3) / 3 * c1c + Pedge_h - 1 - 2 * Rc
    yes = (cal1r <= c1r) & (c1c < math.sqrt(3) * Rc)
    tip1r = c1r[yes]
    tip1c = c1c[yes]

    return tip1r,tip1c


prism_surf = prismsurface(Pedge_h,Pedge,dthick)
face_proj = face_prob(apr_r_Rmin)
tag_prob = tag_kernel(Rmax,Rmin)
pr,pc = tip_edge_contour_idx(Pedge_h,Pedge,dthick)







