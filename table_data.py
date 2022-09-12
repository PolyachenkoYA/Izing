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

OP_interfaces_table[('CS', 11, 0.0, 20, 4, 100, 'optimal')] = np.array([4, 5, 7, 8, 9, 11, 13, 15, 18, 21, 24, 27, 31, 36, 40, 45, 51, 58, 69, 100], dtype=int)
#OP_interfaces_table[('CS', 11, 0.01, 20, 4, 100, 'optimal')] = np.array([4, 6, 7, 9, 11, 15, 18, 22, 25, 28, 31, 35, 38, 41, 45, 49, 54, 60, 69, 100], dtype=int)
OP_interfaces_table[('CS', 11, 0.0, 20, 5, 105, 'optimal')] = np.array([5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 19, 23, 26, 31, 36, 41, 47, 55, 67, 105], dtype=int)
#OP_interfaces_table[('CS', 11, 0.01, 20, 5, 105, 'optimal')] = np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 31, 34, 37, 41, 45, 51, 58, 68, 116], dtype=int)
OP_interfaces_table[('CS', 11, 0.0, 20, 7, 101, 'optimal')] = np.array([7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 21, 25, 29, 33, 37, 43, 49, 57, 69, 101], dtype=int)
#OP_interfaces_table[('CS', 11, 0.01, 20, 7, 101, 'optimal')] = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 34, 37, 40, 43, 47, 52, 58, 68, 114], dtype=int)
OP_interfaces_table[('CS', 10, 0.0, 10, 7, 80, 'optimal')] = np.array([7, 8, 9, 10, 13, 18, 24, 32, 43, 80], dtype=int)
#OP_interfaces_table[('CS', 10, 0.01, 20, 7, 101, 'optimal')] = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 34, 37, 40, 43, 47, 52, 58, 68, 114], dtype=int)
OP_interfaces_table[('CS', 10, 0.0, 10, 20, 80, 'optimal')] = np.array([20, 21, 22, 23, 26, 29, 34, 41, 50, 80], dtype=int)
OP_interfaces_table[('CS', 50, 0.0, 11, 100, 2200, 'spaced')] = np.array([100, 125, 150, 175, 250, 325, 450, 625, 850, 1250, 2200], dtype=int)
OP_interfaces_table[('CS', 30, 0.0, 10, 60, 720, 'spaced')] = np.array([ 60,  80,  100,  120, 140, 160, 216, 288, 387, 720], dtype=int)
OP_interfaces_table[('CS', 20, 0.0, 10, 60, 320, 'spaced')] = np.array([ 60,  80,  100,  120,  140,  160, 180, 200, 320], dtype=int)
OP_interfaces_table[('CS', 10, 0.0, 4, 20, 80, 'spaced')] = np.array([ 20,  40, 60, 80], dtype=int)
OP_interfaces_table[('CS', 10, 0.0, 4, 10, 80, 'spaced')] = np.array([ 10,  30, 50, 80], dtype=int)
OP_interfaces_table[('CS', 10, 0.0, 4, 10, 80, 'optimal')] = np.array([10, 15, 29, 80], dtype=int)
OP_interfaces_table[('CS', 15, 0.0, 10, 20, 185, 'optimal')] = np.array([20, 24, 28, 32, 36, 44, 57, 74, 100, 185], dtype=int)
OP_interfaces_table[('CS', 10, 0.0, 10, 10, 80, 'optimal')] = np.array([10, 11, 12, 14, 18, 23, 28, 36, 47, 80], dtype=int)
OP_interfaces_table[('CS', 100, 0.13, 16, 20, 9500, 'optimal')] = np.array([20, 21, 22, 24, 28, 33, 38, 46, 57, 84, 200, 400, 800, 1600, 3200, 9500], dtype=int)
OP_interfaces_table[('CS', 100, 0.13, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 33, 0.13, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 33, 0.13, 10, 24, 200, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200], dtype=int)

OP_interfaces_table[('CS', 32, 0.15, 8, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 200], dtype=int)
OP_interfaces_table[('CS', 32, 0.15, 8, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 500], dtype=int)

OP_interfaces_table[('CS', 32, 0.13, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 32, 0.13, 10, 24, 200, 'optimal')] = np.array([24, 25, 26, 27, 30, 34, 42, 56, 76, 200], dtype=int)
OP_interfaces_table[('CS', 32, 0.13, 10, 24, 200, 'spaced')] = np.array([24, 47, 67, 87, 108, 128, 148, 168, 200], dtype=int)
OP_interfaces_table[('CS', 32, 0.13, 11, 24, 500, 'spaced')] = np.array([24, 47, 67, 87, 108, 128, 148, 168, 200, 500], dtype=int)
OP_interfaces_table[('CS', 32, 0.13, 9, 24, 200, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 200], dtype=int)
OP_interfaces_table[('CS', 32, 0.13, 9, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 500], dtype=int)
OP_interfaces_table[('CS', 100, 0.13, 9, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 52, 62, 85, 500], dtype=int)

OP_interfaces_table[('CS', 32, 0.12, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 32, 0.12, 10, 24, 220, 'optimal')] = np.array([24, 25, 26, 29, 33, 40, 50, 67, 91, 220], dtype=int)
OP_interfaces_table[('CS', 32, 0.12, 10, 24, 220, 'spaced')] = np.array([24, 28, 48, 68, 88, 108, 128, 148, 168, 220], dtype=int)
OP_interfaces_table[('CS', 32, 0.12, 11, 24, 500, 'spaced')] = np.array([24, 28, 48, 68, 88, 108, 128, 148, 168, 210, 500], dtype=int)
OP_interfaces_table[('CS', 32, 0.12, 9, 24, 220, 'spaced')] = np.array([24, 29, 33, 38, 43, 54, 64, 90, 220], dtype=int)
OP_interfaces_table[('CS', 32, 0.12, 9, 24, 500, 'spaced')] = np.array([24, 29, 34, 39, 44, 54, 70, 95, 500], dtype=int)

OP_interfaces_table[('CS', 32, 0.11, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 32, 0.11, 10, 24, 250, 'optimal')] = np.array([24, 25, 26, 29, 34, 42, 53, 71, 108, 250], dtype=int)
OP_interfaces_table[('CS', 32, 0.11, 10, 24, 250, 'spaced')] = np.array([24, 28, 48, 68, 88, 108, 128, 148, 168, 250], dtype=int)
OP_interfaces_table[('CS', 32, 0.11, 11, 24, 600, 'spaced')] = np.array([24, 28, 48, 68, 88, 108, 128, 148, 168, 220, 600], dtype=int)
OP_interfaces_table[('CS', 32, 0.11, 9, 24, 600, 'spaced')] = np.array([24, 30, 35, 40, 48, 65, 90, 110, 600], dtype=int)

OP_interfaces_table[('CS', 32, 0.10, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 32, 0.10, 10, 24, 350, 'optimal')] = np.array([24, 25, 27, 31, 37, 46, 61, 78, 128, 350], dtype=int)
OP_interfaces_table[('CS', 32, 0.10, 10, 24, 350, 'spaced')] = np.array([24, 29, 49, 69, 89, 109, 129, 149, 169, 350], dtype=int)
OP_interfaces_table[('CS', 32, 0.10, 11, 24, 700, 'spaced')] = np.array([24, 29, 49, 69, 89, 109, 129, 149, 169, 230, 700], dtype=int)
OP_interfaces_table[('CS', 32, 0.10, 9, 24, 700, 'spaced')] = np.array([24, 31, 36, 41, 50, 70, 115, 135, 700], dtype=int)

OP_interfaces_table[('CS', 32, 0.09, 11, 24, 500, 'optimal')] = np.array([24, 25, 26, 28, 32, 37, 42, 50, 84, 200, 500], dtype=int)
OP_interfaces_table[('CS', 32, 0.09, 10, 24, 500, 'optimal')] = np.array([24, 25, 27, 32, 40, 52, 68, 88, 144, 500], dtype=int)
OP_interfaces_table[('CS', 32, 0.09, 10, 24, 500, 'spaced')] = np.array([24, 30, 50, 70, 90, 110, 130, 150, 170, 500], dtype=int)
OP_interfaces_table[('CS', 32, 0.09, 11, 24, 800, 'spaced')] = np.array([24, 30, 50, 70, 90, 110, 130, 150, 170, 240, 800], dtype=int)
OP_interfaces_table[('CS', 32, 0.09, 9, 24, 800, 'spaced')] = np.array([24, 32, 38, 44, 60, 100, 145, 165, 800], dtype=int)

L_reference = 100

Nc_reference_data = {}
# [0, :] - h/J; [1, :] - Nc
Nc_reference_data['1.9'] = np.array([[0.015, 0.017, 0.022, 0.025, 0.030, 0.035], \
									 [1661, 1287, 800, 630, 454, 345]])
Nc_reference_data['1.5'] = np.array([[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13], \
									 [769, 495, 351, 263, 204, 163, 134, 111, 95, 84]])
Nc_reference_data['1.0'] = np.array([[0.05, 0.06, 0.07, 0.08, 0.09, 0.10], \
									 [1014, 703, 519, 397, 319, 258]])

k_AB_reference_data = {}
# [0, :] - h/J; [1, :] - k_AB [1/step]
k_AB_reference_data['1.9'] = np.array([[0.015, 0.017, 0.022, 0.025, 0.030, 0.035], \
									[2.70640180e-17, 1.94871083e-15, 2.78150181e-13, 2.57101146e-12, 3.34593182e-11, 2.60643890e-10]])   # 1/step
k_AB_reference_data['1.5'] = np.array([[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13], \
									[1.02074040e-22, 2.24985501e-19, 6.36595111e-17, 3.25559232e-15, 7.07824687e-14, 1.29696182e-12, 8.51458425e-12, 3.34593182e-11, 1.85122471e-10, 3.66974560e-10]])   # 1/step
k_AB_reference_data['1.2'] = np.array([[0.05, 0.06, 0.07, 0.08, 0.09, 0.10], \
									[2.17418097e-49, 6.93442414e-42, 2.18163240e-36, 1.13107611e-32, 1.25763390e-29, 5.94491455e-27]])   # 1/step

