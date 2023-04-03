import numpy as np

OP_interfaces_table = {}

OP_interfaces_table[('CS', 11, 0.01, 20, 4, 100, 'optimal')] = np.array([4, 5, 7, 8, 9, 11, 13, 15, 18, 21, 24, 27, 31, 36, 40, 45, 51, 58, 69, 100], dtype=int)
#OP_interfaces_table[('CS', 11, -0.01, 20, 4, 100, 'optimal')] = np.array([4, 6, 7, 9, 11, 15, 18, 22, 25, 28, 31, 35, 38, 41, 45, 49, 54, 60, 69, 100], dtype=int)

OP_interfaces_table[('M', 11, -0.01, 20, -117, 117, 'optimal')] = np.array([-117, -115, -113, -109, -105, -97, -93, -85, -81, -75, -69, -61, -53, -47, -37, -29, -17, -3, 19, 117], dtype=int)
OP_interfaces_table[('M', 11, 0.01, 20, -117, 117, 'optimal')] = np.array([-117, -115, -113, -109, -105, -101, -95, -89, -85, -77, -73, -65, -59, -51, -43, -33, -23, -9, 11, 117], dtype=int)
OP_interfaces_table[('M', 11, -0.01, 30, -117, 117, 'optimal')] = np.array([-117, -115, -113, -111, -109, -107, -105, -103, -99, -95, -91, -85, -81, -75, -71, -65, -59, -53, -47, -41, -35, -29, -23, -15,  -7,  1, 11, 27, 51, 117], dtype=int)
OP_interfaces_table[('M', 11, 0.01, 30, -117, 117, 'optimal')] = np.array([-117, -115, -113, -111, -109, -107, -105, -101, -99, -95, -91, -87, -83, -77, -73, -69, -63, -59, -53, -49, -43, -37, -31, -25, -17, -9,  1, 15, 49, 117], dtype=int)

OP_interfaces_table[('M', 11, -0.01, 10, -101, 101, 'optimal')] = np.array([-101, -97, -93, -89, -79, -69, -53, -35, -9, 101], dtype=int)
OP_interfaces_table[('M', 11, 0.01, 10, -101, 101, 'optimal')] = np.array([-101, -97, -93, -89, -81, -69, -53, -37, -11, 101], dtype=int)

OP_interfaces_table[('M', 11, -0.01, 13, -101, 101, 'extra')] = np.array([-101, -97, -93, -89, -79, -69, -53, -35, -9, -3, 3, 9, 101], dtype=int)
OP_interfaces_table[('M', 11, 0.01, 13, -101, 101, 'extra')] = np.array([-101, -97, -93, -89, -81, -69, -53, -37, -9, -3, 3, 9, 101], dtype=int)

OP_interfaces_table[('M', 11, -0.01, 15, -101, 101, 'optimal')] = np.array([-101, -97, -95, -93, -89, -87, -81, -73, -67, -57, -47, -35, -21, 3, 101], dtype=int)
OP_interfaces_table[('M', 11, 0.01, 15, -101, 101, 'optimal')] = np.array([-101, -97, -95, -93, -89, -85, -81, -73, -65, -57, -45, -33, -17, 5, 101], dtype=int)
OP_interfaces_table[('M', 11, -0.01, 18, -101, 101, 'extra')] = np.array([-101, -97, -95, -93, -89, -87, -81, -73, -67, -57, -47, -35, -19, -9, -3, 3, 9, 101], dtype=int)
OP_interfaces_table[('M', 11, 0.01, 18, -101, 101, 'extra')] = np.array([-101, -97, -95, -93, -89, -85, -81, -73, -65, -57, -45, -33, -17, -9, -3, 3, 9, 101], dtype=int)

OP_interfaces_table[('M', 10, -0.01, 18, -80, 80, 'optimal')] = np.array([-80, -78, -76, -74, -72, -70, -68, -64, -60, -56, -52, -44, -36, -28, -20, -8, 8, 80], dtype=int)
OP_interfaces_table[('M', 10, 0.01, 18, -80, 80, 'optimal')] = np.array([-80, -78, -76, -74, -72, -70, -68, -64, -60, -56, -52, -44, -38, -30, -20, -8, 8, 80], dtype=int)
OP_interfaces_table[('M', 10, -0.01, 10, -80, 80, 'optimal')] = np.array([-80, -76, -74, -68, -64, -54, -42, -28, -4, 80], dtype=int)

OP_interfaces_table[('M', 10, -0.01, 19, -80, 80, 'extra')] = np.array([-80, -78, -76, -74, -72, -70, -68, -64, -60, -56, -52, -44, -36, -28, -20, -8, 0, 8, 80], dtype=int)
OP_interfaces_table[('M', 10, 0.01, 19, -80, 80, 'extra')] = np.array([-80, -78, -76, -74, -72, -70, -68, -64, -60, -56, -52, -44, -38, -30, -20, -8, 0, 8, 80], dtype=int)
OP_interfaces_table[('M', 10, -0.01, 9, -80, 80, 'spaced')] = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80], dtype=int)

OP_interfaces_table[('M', 11, -0.01, 20, -101, 101, 'optimal')] = np.array([-101, -99, -97, -95, -93, -91, -89, -85, -81, -77, -73, -65, -59, -51, -43, -33, -23, -9, 11, 101], dtype=int)
OP_interfaces_table[('M', 11, 0.01, 20, -101, 101, 'optimal')] = np.array([-101, -99, -97, -95, -93, -91, -89, -87, -83, -77, -73, -69, -61, -53, -45, -37, -25, -13, 11, 101], dtype=int)
OP_interfaces_table[('M', 11, -0.01, 20, -91, 91, 'optimal')] = np.array([-91, -89, -87, -85, -83, -81, -79, -77, -73, -69, -63, -59, -53, -45, -37, -27, -17, -3, 15, 91], dtype=int)
OP_interfaces_table[('M', 11, 0.01, 20, -91, 91, 'optimal')] = np.array([-91, -89, -87, -85, -83, -81, -79, -77, -75, -71, -67, -63, -55, -51, -43, -33, -23, -9, 11, 91], dtype=int)

OP_interfaces_table[('M', 11, -0.01, 22, -101, 101, 'extra')] = np.array([-101, -99, -97, -95, -93, -91, -89, -85, -81, -77, -73, -65, -59, -51, -43, -33, -23, -13, -5, 5, 13, 101], dtype=int)
OP_interfaces_table[('M', 11, 0.01, 22, -101, 101, 'extra')] = np.array([-101, -99, -97, -95, -93, -91, -89, -87, -83, -77, -73, -69, -61, -53, -45, -37, -25, -13, -5, 5, 13, 101], dtype=int)

OP_interfaces_table[('M', 1.5, 24, 0.13, 9, 24, 500, 'spaced')] = np.array([24, 28, 36, 44, 54, 68, 94, 136, 500], dtype=int)
OP_interfaces_table[('M', 1.5, 18, 0.13, 9, 24, 300, 'spaced')] = np.array([24, 28, 36, 44, 54, 68, 94, 136, 300], dtype=int)

OP_interfaces_table[('M', 1.5, 18, 0.13, 10, 50, 500, 'spaced')] = np.array([30, 34, 42, 50, 60, 74, 100, 142, 300], dtype=int)
OP_interfaces_table[('M', 1.5, 24, 0.13, 10, 50, 500, 'spaced')] = np.array([50, 54, 62, 70, 80, 94, 120, 144, 200, 500], dtype=int)
OP_interfaces_table[('M', 1.5, 50, 0.13, 10, 100, 1000, 'spaced')] = np.array([100, 104, 108, 118, 128, 144, 164, 198, 232, 1000], dtype=int)
OP_interfaces_table[('M', 1.5, 100, 0.13, 10, 280, 1000, 'spaced')] = np.array([280, 284, 288, 296, 308, 324, 344, 370, 440, 1000], dtype=int)

OP_interfaces_table[('M', 1.5, 18, 0.15, 10, 30, 300, 'spaced')] = np.array([30, 34, 36, 42, 46, 54, 66, 84, 112, 300], dtype=int)
OP_interfaces_table[('M', 1.5, 24, 0.15, 10, 50, 500, 'spaced')] = np.array([50, 54, 56, 62, 66, 74, 86, 104, 132, 500], dtype=int)
OP_interfaces_table[('M', 1.5, 50, 0.15, 10, 100, 1000, 'spaced')] = np.array([100, 104, 108, 114, 122, 132, 144, 160, 186, 1000], dtype=int)
OP_interfaces_table[('M', 1.5, 100, 0.15, 10, 280, 1000, 'spaced')] = np.array([280, 284, 288, 296, 308, 324, 344, 370, 440, 1000], dtype=int)

OP_interfaces_table[('CS', 1.5, 11, 0.0, 20, 4, 100, 'optimal')] = np.array([4, 5, 7, 8, 9, 11, 13, 15, 18, 21, 24, 27, 31, 36, 40, 45, 51, 58, 69, 100], dtype=int)
#OP_interfaces_table[('CS', 1.5, 11, 0.01, 20, 4, 100, 'optimal')] = np.array([4, 6, 7, 9, 11, 15, 18, 22, 25, 28, 31, 35, 38, 41, 45, 49, 54, 60, 69, 100], dtype=int)
OP_interfaces_table[('CS', 1.5, 11, 0.0, 20, 5, 105, 'optimal')] = np.array([5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 19, 23, 26, 31, 36, 41, 47, 55, 67, 105], dtype=int)
#OP_interfaces_table[('CS', 1.5, 11, 0.01, 20, 5, 105, 'optimal')] = np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 31, 34, 37, 41, 45, 51, 58, 68, 116], dtype=int)
OP_interfaces_table[('CS', 1.5, 11, 0.0, 20, 7, 101, 'optimal')] = np.array([7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 21, 25, 29, 33, 37, 43, 49, 57, 69, 101], dtype=int)
#OP_interfaces_table[('CS', 1.5, 11, 0.01, 20, 7, 101, 'optimal')] = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 34, 37, 40, 43, 47, 52, 58, 68, 114], dtype=int)
OP_interfaces_table[('CS', 1.5, 10, 0.0, 10, 7, 80, 'optimal')] = np.array([7, 8, 9, 10, 13, 18, 24, 32, 43, 80], dtype=int)
#OP_interfaces_table[('CS', 1.5, 10, 0.01, 20, 7, 101, 'optimal')] = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 34, 37, 40, 43, 47, 52, 58, 68, 114], dtype=int)
OP_interfaces_table[('CS', 1.5, 10, 0.0, 10, 20, 80, 'optimal')] = np.array([20, 21, 22, 23, 26, 29, 34, 41, 50, 80], dtype=int)
OP_interfaces_table[('CS', 1.5, 50, 0.0, 11, 100, 2200, 'spaced')] = np.array([100, 125, 150, 175, 250, 325, 450, 625, 850, 1250, 2200], dtype=int)
OP_interfaces_table[('CS', 1.5, 30, 0.0, 10, 60, 720, 'spaced')] = np.array([ 60,  80,  100,  120, 140, 160, 216, 288, 387, 720], dtype=int)
OP_interfaces_table[('CS', 1.5, 20, 0.0, 10, 60, 320, 'spaced')] = np.array([ 60,  80,  100,  120,  140,  160, 180, 200, 320], dtype=int)
OP_interfaces_table[('CS', 1.5, 10, 0.0, 4, 20, 80, 'spaced')] = np.array([ 20,  40, 60, 80], dtype=int)
OP_interfaces_table[('CS', 1.5, 10, 0.0, 4, 10, 80, 'spaced')] = np.array([ 10,  30, 50, 80], dtype=int)
OP_interfaces_table[('CS', 1.5, 10, 0.0, 4, 10, 80, 'optimal')] = np.array([10, 15, 29, 80], dtype=int)
OP_interfaces_table[('CS', 1.5, 15, 0.0, 10, 20, 185, 'optimal')] = np.array([20, 24, 28, 32, 36, 44, 57, 74, 100, 185], dtype=int)
OP_interfaces_table[('CS', 1.5, 10, 0.0, 10, 10, 80, 'optimal')] = np.array([10, 11, 12, 14, 18, 23, 28, 36, 47, 80], dtype=int)
OP_interfaces_table[('CS', 1.5, 100, 0.13, 16, 20, 9500, 'optimal')] = np.array([20, 21, 22, 24, 28, 33, 38, 46, 57, 84, 200, 400, 800, 1600, 3200, 9500], dtype=int)
OP_interfaces_table[('CS', 1.5, 100, 0.13, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 33, 0.13, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 33, 0.13, 10, 24, 200, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200], dtype=int)

OP_interfaces_table[('CS', 1.5, 32, 0.15, 8, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.15, 8, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 500], dtype=int)

OP_interfaces_table[('CS', 1.5, 100, 0.15, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 78, 0.15, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 64, 0.15, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 56, 0.15, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 48, 0.15, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.15, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 24, 0.15, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 18, 0.15, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)

OP_interfaces_table[('CS', 1.5, 64, 0.15, 8, 29, 200, 'spaced')] = np.array([29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.15, 8, 29, 200, 'spaced')] = np.array([29, 34, 39, 44, 52, 62, 85, 200], dtype=int)

OP_interfaces_table[('CS', 1.5, 100, 0.15, 15, 24, 210, 'spaced')] = np.array([24,32,40,48,60,72,84,96,110,125,140,155,170,190,210], dtype=int)
OP_interfaces_table[('CS', 1.5, 64, 0.15, 15, 24, 210, 'spaced')] = np.array([24,32,40,48,60,72,84,96,110,125,140,155,170,190,210], dtype=int)
OP_interfaces_table[('CS', 1.5, 48, 0.15, 15, 24, 210, 'spaced')] = np.array([24,32,40,48,60,72,84,96,110,125,140,155,170,190,210], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.15, 15, 24, 210, 'spaced')] = np.array([24,32,40,48,60,72,84,96,110,125,140,155,170,190,210], dtype=int)

OP_interfaces_table[('CS', 1.5, 100, 0.15, 8, 24, 200, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 72, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 64, 0.15, 8, 24, 200, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 72, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 48, 0.15, 8, 24, 200, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 72, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.15, 8, 24, 200, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 72, 200], dtype=int)

OP_interfaces_table[('CS', 1.5, 100, 0.15, 14, 32, 210, 'spaced')] = np.array([32,40,48,60,72,84,96,110,125,140,155,170,190,210], dtype=int)
OP_interfaces_table[('CS', 1.5, 64, 0.15, 14, 32, 210, 'spaced')] = np.array([32,40,48,60,72,84,96,110,125,140,155,170,190,210], dtype=int)
OP_interfaces_table[('CS', 1.5, 48, 0.15, 14, 32, 210, 'spaced')] = np.array([32,40,48,60,72,84,96,110,125,140,155,170,190,210], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.15, 14, 32, 210, 'spaced')] = np.array([32,40,48,60,72,84,96,110,125,140,155,170,190,210], dtype=int)

OP_interfaces_table[('CS', 1.5, 100, 0.13, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 78, 0.13, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 56, 0.13, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 42, 0.13, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.13, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 24, 0.13, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 18, 0.13, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 10, 0.13, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)

OP_interfaces_table[('CS', 1.5, 32, 0.13, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.13, 10, 24, 200, 'optimal')] = np.array([24, 25, 26, 27, 30, 34, 42, 56, 76, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.13, 10, 24, 200, 'spaced')] = np.array([24, 47, 67, 87, 108, 128, 148, 168, 200], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.13, 11, 24, 500, 'spaced')] = np.array([24, 47, 67, 87, 108, 128, 148, 168, 200, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.13, 9, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 100, 0.13, 9, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 18, 0.13, 9, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 10, 0.13, 9, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 24, 0.13, 9, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 500], dtype=int)

OP_interfaces_table[('CS', 1.5, 32, 0.12, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.12, 10, 24, 220, 'optimal')] = np.array([24, 25, 26, 29, 33, 40, 50, 67, 91, 220], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.12, 10, 24, 220, 'spaced')] = np.array([24, 28, 48, 68, 88, 108, 128, 148, 168, 220], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.12, 11, 24, 500, 'spaced')] = np.array([24, 28, 48, 68, 88, 108, 128, 148, 168, 210, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.12, 9, 24, 220, 'spaced')] = np.array([24, 29, 33, 38, 43, 54, 64, 90, 220], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.12, 9, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 54, 70, 95, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.12, 9, 24, 250, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 72, 95, 250], dtype=int)

OP_interfaces_table[('CS', 1.5, 32, 0.11, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.11, 10, 24, 250, 'optimal')] = np.array([24, 25, 26, 29, 34, 42, 53, 71, 108, 250], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.11, 10, 24, 250, 'spaced')] = np.array([24, 28, 48, 68, 88, 108, 128, 148, 168, 250], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.11, 11, 24, 600, 'spaced')] = np.array([24, 28, 48, 68, 88, 108, 128, 148, 168, 220, 600], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.11, 9, 24, 600, 'spaced')] = np.array([24, 30, 35, 40, 48, 65, 90, 110, 600], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.11, 9, 24, 300, 'spaced')] = np.array([24, 32, 40, 48, 58, 70, 90, 110, 300], dtype=int)
#OP_interfaces_table[('CS', 1.5, 32, 0.11, 9, 24, 200, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 90, 110, 200], dtype=int)

OP_interfaces_table[('CS', 1.5, 32, 0.10, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.10, 10, 24, 350, 'optimal')] = np.array([24, 25, 27, 31, 37, 46, 61, 78, 128, 350], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.10, 10, 24, 350, 'spaced')] = np.array([24, 29, 49, 69, 89, 109, 129, 149, 169, 350], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.10, 11, 24, 700, 'spaced')] = np.array([24, 29, 49, 69, 89, 109, 129, 149, 169, 230, 700], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.10, 9, 24, 700, 'spaced')] = np.array([24, 31, 36, 41, 50, 70, 115, 135, 700], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.10, 9, 24, 350, 'spaced')] = np.array([24, 32, 40, 48, 56, 70, 95, 135, 350], dtype=int)

OP_interfaces_table[('CS', 1.5, 32, 0.09, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.09, 10, 24, 500, 'optimal')] = np.array([24, 25, 27, 32, 40, 52, 68, 88, 144, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.09, 10, 24, 500, 'spaced')] = np.array([24, 30, 50, 70, 90, 110, 130, 150, 170, 500], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.09, 11, 24, 800, 'spaced')] = np.array([24, 30, 50, 70, 90, 110, 130, 150, 170, 240, 800], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.09, 9, 24, 800, 'spaced')] = np.array([24, 32, 38, 44, 60, 100, 145, 165, 800], dtype=int)
OP_interfaces_table[('CS', 1.5, 32, 0.09, 9, 24, 400, 'spaced')] = np.array([24, 32, 40, 48, 60, 100, 145, 165, 400], dtype=int)
#OP_interfaces_table[('CS', 1.5, 32, 0.09, 9, 24, 200, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 72, 200], dtype=int)

OP_interfaces_table[('CS', 1.5, 32, 0.08, 9, 24, 500, 'spaced')] = np.array([24, 32, 40, 48, 60, 100, 145, 200, 500], dtype=int)

OP_interfaces_table[('CS', 1.9, 32, 0.09, 9, 24, 300, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 74, 86, 300], dtype=int)
OP_interfaces_table[('CS', 1.9, 32, 0.08, 9, 24, 350, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 74, 95, 350], dtype=int)
OP_interfaces_table[('CS', 1.9, 32, 0.07, 9, 24, 400, 'spaced')] = np.array([24, 32, 40, 48, 58, 70, 90, 110, 400], dtype=int)
OP_interfaces_table[('CS', 1.9, 32, 0.06, 9, 24, 450, 'spaced')] = np.array([24, 32, 40, 48, 56, 70, 95, 135, 450], dtype=int)
OP_interfaces_table[('CS', 1.9, 32, 0.05, 9, 24, 500, 'spaced')] = np.array([24, 32, 40, 48, 60, 100, 145, 165, 500], dtype=int)
OP_interfaces_table[('CS', 1.9, 32, 0.04, 9, 24, 600, 'spaced')] = np.array([24, 32, 40, 48, 60, 100, 145, 200, 600], dtype=int)

OP_interfaces_table[('CS', 2.0, 32, 0.09, 9, 24, 300, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 74, 86, 300], dtype=int)
OP_interfaces_table[('CS', 2.0, 32, 0.08, 9, 24, 350, 'spaced')] = np.array([24, 32, 40, 48, 56, 64, 72, 80, 350], dtype=int)
OP_interfaces_table[('CS', 2.0, 32, 0.07, 9, 24, 400, 'spaced')] = np.array([24, 32, 40, 48, 56, 78, 80, 100, 400], dtype=int)
OP_interfaces_table[('CS', 2.0, 32, 0.06, 9, 24, 450, 'spaced')] = np.array([24, 32, 40, 48, 56, 70, 95, 135, 450], dtype=int)
OP_interfaces_table[('CS', 2.0, 32, 0.05, 9, 24, 500, 'spaced')] = np.array([24, 32, 40, 48, 60, 75, 100, 145, 500], dtype=int)
OP_interfaces_table[('CS', 2.0, 32, 0.04, 9, 24, 600, 'spaced')] = np.array([24, 32, 40, 48, 60, 100, 145, 200, 600], dtype=int)

OP_interfaces_table['hJ0.15'] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int) # h/J = 0.15
OP_interfaces_table['hJ0.13'] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 500], dtype=int) # h/J = 0.13
OP_interfaces_table['hJ0.12'] = np.array([24, 29, 34, 39, 44, 54, 70, 95, 500], dtype=int) # h/J = 0.12
OP_interfaces_table['hJ0.1'] = np.array([24, 32, 40, 48, 56, 70, 95, 135, 350], dtype=int) # h/J = 0.1

OP_interfaces_table['hJ0.13_2'] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table['hJ0.12_2'] = np.array([24, 32, 40, 48, 56, 64, 72, 95, 250], dtype=int)
OP_interfaces_table['hJ0.11_2'] = np.array([24, 32, 40, 48, 58, 70, 90, 110, 300], dtype=int)
OP_interfaces_table['hJ0.10_2'] = np.array([24, 32, 40, 48, 56, 70, 95, 135, 350], dtype=int)
OP_interfaces_table['hJ0.09_2'] = np.array([24, 32, 40, 48, 60, 100, 145, 165, 400], dtype=int)
OP_interfaces_table['hJ0.08_2'] = np.array([24, 32, 40, 48, 60, 100, 145, 200, 500], dtype=int)

OP_interfaces_table['mu1'] = np.array([24, 29, 34, 39, 44, 52, 62, 85,    200], dtype=int)
OP_interfaces_table['mu2'] = np.array([10, 15, 20, 25, 30, 35, 40, 45, 52, 85], dtype=int)
OP_interfaces_table['mu3'] = np.array([24, 29, 34, 39, 44, 52, 62], dtype=int)
OP_interfaces_table['mu4'] = np.array([24, 29, 34, 39, 44, 46, 48, 50, 52, 54], dtype=int)
OP_interfaces_table['mu5'] = np.array([20, 28, 36, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table['mu6'] = np.array([12, 20, 28, 36, 44, 52, 62, 85, 200], dtype=int)

OP_interfaces_table['nvt1'] = np.array([15, 20, 24, 29, 34, 39, 44, 46, 48, 50, 52, 54], dtype=int)
OP_interfaces_table['nvt2'] = np.array([12, 16, 20, 24, 29, 34, 39, 44, 50, 60], dtype=int)
OP_interfaces_table['nvt3'] = np.array([10, 15, 20, 25, 30, 35, 40, 45, 52, 62, 85,    200], dtype=int)
OP_interfaces_table['nvt4'] = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 52, 62, 85,    200], dtype=int)
OP_interfaces_table['nvt5'] = np.array([10, 15, 20, 25, 30, 35, 40], dtype=int)
OP_interfaces_table['nvt6'] = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=int)
OP_interfaces_table['nvt7'] = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 40], dtype=int)
OP_interfaces_table['nvt8'] = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 40], dtype=int)
OP_interfaces_table['nvt9'] = np.array([11, 13, 15, 17, 19, 21, 23, 25], dtype=int)
OP_interfaces_table['nvt10'] = np.array([24, 29, 34, 39, 44, 52, 62, 85], dtype=int)
OP_interfaces_table['nvt11'] = np.array([14, 19, 24, 29, 34, 39, 44], dtype=int)
OP_interfaces_table['nvt12'] = np.array([13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 40, 45], dtype=int)

L_reference = 100

Nc_reference_data = {}
# [0, :] - h/J; [1, :] - Nc
Nc_reference_data['1.9'] = np.array([[0.015, 0.017, 0.022, 0.025, 0.030, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], \
									 [1661, 1287, 800, 630, 454, 345, 243, 182, 135, 105, 85, 71]])
Nc_reference_data['1.5'] = np.array([[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13], \
									 [769, 495, 351, 263, 204, 163, 134, 111, 95, 84]])
Nc_reference_data['1.0'] = np.array([[0.05, 0.06, 0.07, 0.08, 0.09, 0.10], \
									 [1014, 703, 519, 397, 319, 258]])
									 
my_data = np.array([[0.13, 82.88576154555986], \
[0.12, 95.10592548880943], \
[0.11, 112.49328565207617], \
[0.10, 133.84760914921972], \
[0.09, 162.7877585176651], \
[0.08, 203.91906104455563]]).T

k_AB_reference_data = {}
# [0, :] - h/J; [1, :] - k_AB [1/step]
k_AB_reference_data['1.9'] = np.array([[0.015, 0.017, 0.022, 0.025, 0.030, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], \
									[2.70640180e-17, 1.94871083e-15, 2.78150181e-13, 2.57101146e-12, 3.34593182e-11, 2.60643890e-10, 9.18001299e-10, 9.21422482e-09, 4.23464928e-08, 1.49999727e-07,
       3.94568834e-07, 7.99962201e-07]])   # 1/step
k_AB_reference_data['1.5'] = np.array([[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13], \
									[1.02074040e-22, 2.24985501e-19, 6.36595111e-17, 3.25559232e-15, 7.07824687e-14, 1.29696182e-12, 8.51458425e-12, 3.34593182e-11, 1.85122471e-10, 3.66974560e-10]])   # 1/step
k_AB_reference_data['1.2'] = np.array([[0.05, 0.06, 0.07, 0.08, 0.09, 0.10], \
									[2.17418097e-49, 6.93442414e-42, 2.18163240e-36, 1.13107611e-32, 1.25763390e-29, 5.94491455e-27]])   # 1/step

mc_crytical = {}
mc_crytical['1.5'] = {}
mc_crytical['1.5']['0.13'] = np.array([[24, 50, 100], \
									   [0.7, 0.91, 0.955]])

qiwei_data = np.array([[0.2, 1.1535516295814618], \
[0.19, 2.386589786858581], \
[0.18, 3.0692837285991494], \
[0.17, 3.4732875590858057], \
[0.16, 3.782489906389384], \
[0.15, 4.092955633477125], \
[0.14, 4.298662347082275], \
[0.13, 4.524343346616744], \
[0.12, 4.681342002991099], \
[0.11, 4.864486887188456], \
[0.10, 4.969312358864733], \
[0.09, 5.230202675339295], \
[0.08, 5.320175096324757], \
[0.07, 5.469693148533981], \
[0.06, 5.8061585213899916], \
[0.05, 6.046185695783258], \
[0.04, 6.71199851680717], \
[0.03, 7.595485789666263], \
[0.02, 9.789054371900612], \
[0.01, 15.984671064343198]]).T
qiwei_J_d_Jc = 2.27/1.5
qiwei_data[1, :] = np.pi * qiwei_data[1, :]**2   # these are radia of circle nuclei

sgm_analyt = np.array([[0.015604681404421283, 2.256006006006006], \
[0.04421326397919373, 2.256006006006006], \
[0.07022106631989594, 2.2522522522522523], \
[0.09622886866059815, 2.244744744744745], \
[0.130039011703511, 2.2484984984984986], \
[0.15864759427828357, 2.2409909909909906], \
[0.1924577373211963, 2.2409909909909906], \
[0.23146944083224963, 2.22972972972973], \
[0.27048114434330306, 2.225975975975976], \
[0.31469440832249673, 2.218468468468468], \
[0.35630689206762034, 2.203453453453453], \
[0.40832249674902477, 2.184684684684685], \
[0.447334200260078, 2.1659159159159156], \
[0.4967490247074122, 2.1471471471471473], \
[0.540962288686606, 2.128378378378378], \
[0.5929778933680104, 2.1021021021021022], \
[0.6397919375812745, 2.0720720720720722], \
[0.7048114434330298, 2.0307807807807805], \
[0.7672301690507152, 1.981981981981982], \
[0.8426527958387516, 1.9181681681681682], \
[0.9206762028608583, 1.8543543543543544], \
[0.9934980494148246, 1.7830330330330328], \
[1.0611183355006504, 1.7192192192192193], \
[1.1157347204161248, 1.6629129129129128], \
[1.1625487646293888, 1.6066066066066067], \
[1.2145643693107933, 1.5540540540540542], \
[1.2769830949284788, 1.4827327327327329], \
[1.3420026007802344, 1.4039039039039038], \
[1.40702210663199, 1.325075075075075], \
[1.4616384915474643, 1.2575075075075073], \
[1.5084525357607284, 1.1936936936936937], \
[1.5552665799739924, 1.1336336336336335], \
[1.6020806241872565, 1.0698198198198199], \
[1.6514954486345905, 1.0022522522522521], \
[1.7009102730819246, 0.9346846846846846], \
[1.7555266579973994, 0.8558558558558558], \
[1.8075422626788038, 0.7807807807807807], \
[1.8647594278283486, 0.6869369369369367], \
[1.9245773732119638, 0.6006006006006008], \
[1.9765929778933686, 0.5142642642642641], \
[2.0364109232769834, 0.4204204204204207], \
[2.088426527958388, 0.3303303303303302], \
[2.137841352405722, 0.2515015015015014], \
[2.1872561768530563, 0.1689189189189193], \
[2.2262678803641096, 0.09384384384384381], \
[2.2600780234070226, 0.018768768768769206]])
