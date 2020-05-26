import numpy as np

from visual_propagation_ranges import get_boundary_value, iterativily_find_size


def test_ranges():
    x = np.reshape([0., 1., 2., 2.5, 3.5], (-1, 1))
    y = [0, 0, 1, 0, 1]
    # clf = SVC(gamma='auto', decision_function_shape='ovo', kernel='linear')
    # clf.fit(x, y)
    # scale = np.reshape(np.linspace(0, 4, 500), (-1, 1))
    # boundary = clf.decision_function(scale)
    # w_norm = np.linalg.norm(clf.coef_)
    # dist = boundary / w_norm
    # plt.plot(scale, dist)
    # plt.show()
    #
    # pivot_point = -1 * clf.decision_function(np.reshape([0.], (-1,1)))[0] / w_norm
    # print(pivot_point)
    # assert  pivot_point == 2.2499999999999996
    size = get_boundary_value(x, y)
    assert size == 2.2499999999999996

def test_ranges_its():
    l = [0.,1.,2.]
    u = [2.5,3.5]
    size = iterativily_find_size(l, u)
    assert size == 2.25

def test_ranges_its2():
    l, _ = zip(*[(0.0024500000000000004, 0), (0.0039000000000000003, 0), (0.006500000000000001, 0), (0.0079, 0),
            (0.023899999999999998, 0), (0.025, 0), (0.025, 0), (0.0254, 0), (0.0254, 0), (0.026000000000000002, 0),
            (0.05, 0), (0.05, 0), (0.0508, 0), (0.0508, 0), (0.0508, 0), (0.065, 0), (0.08128, 0), (0.08509, 0),
            (0.1016, 0), (0.10300000000000001, 0), (0.11, 0), (0.11669999999999998, 0), (0.11696, 0), (0.127, 0),
            (0.1397, 0), (0.15239999999999998, 0), (0.15239999999999998, 0), (0.17779999999999999, 0),
            (0.17779999999999999, 0), (0.1905, 0), (0.2032, 0), (0.2032, 0), (0.20466666666666666, 0), (0.215, 0),
            (0.21589999999999998, 0), (0.25, 0), (0.27, 0), (0.2794, 0), (0.30479999999999996, 0), (0.31, 0),
            (0.3175, 0), (0.32144, 0), (0.33, 0), (0.3302, 0), (0.3302, 0), (0.35000000000000003, 0),
            (0.3725333333333333, 0), (0.4064, 0), (0.43179999999999996, 0), (0.43815, 0), (0.4625, 0), (0.5, 0),
            (0.54, 0), (0.5700000000000001, 0), (0.58, 0), (0.6, 0), (0.6, 0), (0.6096, 0), (0.6096, 0), (0.635, 0),
            (0.6476999999999999, 0), (0.8, 0), (0.86, 0), (0.875, 0), (0.9144, 0), (1.0, 0), (1.0668, 0), (1.08, 0),
            (1.122968, 0), (1.165225, 0), (1.2065000000000001, 0), (1.22, 0), (1.4986, 0), (1.524, 0),
            (1.6890999999999998, 0), (1.7018, 0), (1.75, 0), (2.1336, 0), (3.3528000000000002, 0), (3.895, 0),
            (4.34, 0), (4.6, 0), (5.05, 0), (5.627, 0), (6.4, 0), (7.010400000000001, 0), (7.62, 0), (35.24, 0),
            (73.152, 0)])
    l = list(l)
    u , _ = zip(*[(0.04225, 1), (0.05, 1), (0.07, 1), (0.127, 1), (0.1804, 1), (0.1905, 1),
            (0.21589999999999998, 1), (0.24, 1), (0.254, 1), (0.254, 1), (0.2709333333333333, 1),
            (0.27177999999999997, 1), (0.305, 1), (0.34, 1), (0.35559999999999997, 1), (0.37795199999999995, 1),
            (0.381, 1), (0.382625, 1), (0.4, 1), (0.41910000000000003, 1), (0.45, 1), (0.46, 1), (0.5, 1), (0.508, 1),
            (0.51, 1), (0.53, 1), (0.5322, 1), (0.555, 1), (0.58, 1), (0.60145, 1), (0.6716857142857142, 1),
            (0.6912499999999999, 1), (0.7493, 1), (0.762, 1), (0.765, 1), (0.8635999999999999, 1), (0.9144, 1),
            (0.9144000000000001, 1), (0.9144000000000001, 1), (0.943, 1), (1.05, 1), (1.1811, 1),
            (1.1949999999999998, 1), (1.2191999999999998, 1), (1.2191999999999998, 1), (1.2192, 1), (1.2192, 1),
            (1.2361333333333333, 1), (1.2699999999999998, 1), (1.2953999999999999, 1), (1.2954, 1), (1.3144, 1),
            (1.3335, 1), (1.4, 1), (1.4484, 1), (1.4986, 1), (1.5, 1), (1.524, 1), (1.524, 1), (1.5621, 1), (1.5748, 1),
            (1.6, 1), (1.6764, 1), (1.8, 1), (1.801, 1), (1.8288000000000002, 1), (1.8288000000000002, 1),
            (1.8288000000000002, 1), (1.9, 1), (1.9304, 1), (2.1336, 1), (2.2224999999999997, 1),
            (2.289066666666667, 1), (2.3733, 1), (2.435, 1), (2.616, 1), (2.7432, 1), (2.745, 1), (3.048, 1),
            (3.048, 1), (3.048, 1), (3.81, 1), (3.9624000000000006, 1), (4.0, 1), (4.2672, 1), (4.44, 1),
            (4.7395000000000005, 1), (4.784, 1), (4.95, 1), (6.13, 1), (7.315200000000001, 1), (8.3312, 1), (9.144, 1),
            (9.144, 1), (10.38216, 1), (15.0, 1), (18.11816, 1), (19.812, 1), (21.336000000000002, 1),
            (23.270666666666667, 1), (24.0, 1), (34.7472, 1), (35.0, 1), (44.196000000000005, 1), (57.0, 1), (70.0, 1),
            (105.0, 1), (105.156, 1), (152.0, 1), (300.0, 1), (325.0, 1), (375.0, 1), (400.0, 1), (450.0, 1),
            (8181.0, 1), (80000.0, 1), (144837.0, 1), (190000.0, 1)])
    u = list(u)
    size = iterativily_find_size(l, u)
    print(size)



def test_ranges2():
    data = [(0.0024500000000000004, 0), (0.0039000000000000003, 0), (0.006500000000000001, 0), (0.0079, 0),
            (0.023899999999999998, 0), (0.025, 0), (0.025, 0), (0.0254, 0), (0.0254, 0), (0.026000000000000002, 0),
            (0.05, 0), (0.05, 0), (0.0508, 0), (0.0508, 0), (0.0508, 0), (0.065, 0), (0.08128, 0), (0.08509, 0),
            (0.1016, 0), (0.10300000000000001, 0), (0.11, 0), (0.11669999999999998, 0), (0.11696, 0), (0.127, 0),
            (0.1397, 0), (0.15239999999999998, 0), (0.15239999999999998, 0), (0.17779999999999999, 0),
            (0.17779999999999999, 0), (0.1905, 0), (0.2032, 0), (0.2032, 0), (0.20466666666666666, 0), (0.215, 0),
            (0.21589999999999998, 0), (0.25, 0), (0.27, 0), (0.2794, 0), (0.30479999999999996, 0), (0.31, 0),
            (0.3175, 0), (0.32144, 0), (0.33, 0), (0.3302, 0), (0.3302, 0), (0.35000000000000003, 0),
            (0.3725333333333333, 0), (0.4064, 0), (0.43179999999999996, 0), (0.43815, 0), (0.4625, 0), (0.5, 0),
            (0.54, 0), (0.5700000000000001, 0), (0.58, 0), (0.6, 0), (0.6, 0), (0.6096, 0), (0.6096, 0), (0.635, 0),
            (0.6476999999999999, 0), (0.8, 0), (0.86, 0), (0.875, 0), (0.9144, 0), (1.0, 0), (1.0668, 0), (1.08, 0),
            (1.122968, 0), (1.165225, 0), (1.2065000000000001, 0), (1.22, 0), (1.4986, 0), (1.524, 0),
            (1.6890999999999998, 0), (1.7018, 0), (1.75, 0), (2.1336, 0), (3.3528000000000002, 0), (3.895, 0),
            (4.34, 0), (4.6, 0), (5.05, 0), (5.627, 0), (6.4, 0), (7.010400000000001, 0), (7.62, 0), (35.24, 0),
            (73.152, 0), (0.04225, 1), (0.05, 1), (0.07, 1), (0.127, 1), (0.1804, 1), (0.1905, 1),
            (0.21589999999999998, 1), (0.24, 1), (0.254, 1), (0.254, 1), (0.2709333333333333, 1),
            (0.27177999999999997, 1), (0.305, 1), (0.34, 1), (0.35559999999999997, 1), (0.37795199999999995, 1),
            (0.381, 1), (0.382625, 1), (0.4, 1), (0.41910000000000003, 1), (0.45, 1), (0.46, 1), (0.5, 1), (0.508, 1),
            (0.51, 1), (0.53, 1), (0.5322, 1), (0.555, 1), (0.58, 1), (0.60145, 1), (0.6716857142857142, 1),
            (0.6912499999999999, 1), (0.7493, 1), (0.762, 1), (0.765, 1), (0.8635999999999999, 1), (0.9144, 1),
            (0.9144000000000001, 1), (0.9144000000000001, 1), (0.943, 1), (1.05, 1), (1.1811, 1),
            (1.1949999999999998, 1), (1.2191999999999998, 1), (1.2191999999999998, 1), (1.2192, 1), (1.2192, 1),
            (1.2361333333333333, 1), (1.2699999999999998, 1), (1.2953999999999999, 1), (1.2954, 1), (1.3144, 1),
            (1.3335, 1), (1.4, 1), (1.4484, 1), (1.4986, 1), (1.5, 1), (1.524, 1), (1.524, 1), (1.5621, 1), (1.5748, 1),
            (1.6, 1), (1.6764, 1), (1.8, 1), (1.801, 1), (1.8288000000000002, 1), (1.8288000000000002, 1),
            (1.8288000000000002, 1), (1.9, 1), (1.9304, 1), (2.1336, 1), (2.2224999999999997, 1),
            (2.289066666666667, 1), (2.3733, 1), (2.435, 1), (2.616, 1), (2.7432, 1), (2.745, 1), (3.048, 1),
            (3.048, 1), (3.048, 1), (3.81, 1), (3.9624000000000006, 1), (4.0, 1), (4.2672, 1), (4.44, 1),
            (4.7395000000000005, 1), (4.784, 1), (4.95, 1), (6.13, 1), (7.315200000000001, 1), (8.3312, 1), (9.144, 1),
            (9.144, 1), (10.38216, 1), (15.0, 1), (18.11816, 1), (19.812, 1), (21.336000000000002, 1),
            (23.270666666666667, 1), (24.0, 1), (34.7472, 1), (35.0, 1), (44.196000000000005, 1), (57.0, 1), (70.0, 1),
            (105.0, 1), (105.156, 1), (152.0, 1), (300.0, 1), (325.0, 1), (375.0, 1), (400.0, 1), (450.0, 1),
            (8181.0, 1), (80000.0, 1), (144837.0, 1), (190000.0, 1)]
    x, y = zip(*data)
    x = np.reshape(x, (-1, 1))

    pivot_point = get_boundary_value(x, y)
    print(pivot_point)
