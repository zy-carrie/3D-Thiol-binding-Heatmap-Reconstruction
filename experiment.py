import numpy as np
from heatmap import heatmap
import util



def stats(bindingmap):
    tr, tc = util.tip_idx(bindingmap.Pedge_h)
    pr, pc = util.tip_edge_contour_idx(bindingmap.Pedge_h, bindingmap.Pedge, bindingmap.dthick)

    data = bindingmap.bind_prob
    data[:, :, -1] = data[:, :, 0]

    face = np.zeros_like(data)
    edge = np.zeros_like(data)
    tip = np.zeros_like(data)
    face[:, :, :] = data[:, :, :]
    edge[:, :, :] = data[:, :, :]
    edge[:-1, :, :] = 0
    tip[:, :, :] = data[:, :, :]
    tip = data[tr, tc, :]
    face[pr, pc, :] = 0
    print('face', face[:, :, :].sum())
    print('edge', edge.sum())
    print('tip', data[tr, tc, :].sum())

    area_frac = np.array([0.824, 0.146, 0.029])
    prob = np.array([face.sum(), edge.sum(), tip.sum()]) / data.sum()
    prob_dens = prob / area_frac
    enhancement = prob_dens / prob_dens[0]
    print(prob)
    print(prob_dens)
    print(enhancement)


def get_stats3():
    filename3 = 'data/automated position classify3.xlsx'
    bindmap3 = heatmap(filename3)
    bindmap3.add_allEvents()
    stats(bindmap3)

def dataset1():
    filename = 'data/automated position classify1.xlsx'
    bindmap = heatmap(filename)
    bindmap.add_allEvents()
    bindmap.save_bindmap('class1.npy')
    stats(bindmap)
def dataset2():
    filename = 'data/automated position classify2.xlsx'
    bindmap = heatmap(filename)
    bindmap.add_allEvents()
    bindmap.save_bindmap('class2.npy')
    stats(bindmap)

def dataset3():
    filename = 'data/automated position classify3.xlsx'
    bindmap = heatmap(filename)
    bindmap.add_allEvents()
    bindmap.save_bindmap('class3.npy')
    stats(bindmap)

def main():
    filename = 'data/automated position classify.xlsx'
    bindmap = heatmap(filename)
    bindmap.add_allEvents()
    bindmap.save_bindmap('data1003.npy')
    stats(bindmap)

if __name__ == "__main__":
    main()

