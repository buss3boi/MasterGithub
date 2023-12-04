# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:28:12 2023

@author: busse
"""

import numpy as np

np.seterr(divide='ignore', invalid='ignore')

def funk_semivar_mean_var(Z1, Z2, ant, maxdist):
    xobs1, yobs1, zo1 = Z1[:, 0], Z1[:, 1], Z1[:, 2]
    xobs2, yobs2, zo2 = Z2[:, 0], Z2[:, 1], Z2[:, 2]

    xobs1 -= min(xobs1)
    yobs1 -= min(yobs1)
    xobs2 -= min(xobs2)
    yobs2 -= min(yobs2)

    N1 = len(xobs1)
    N2 = len(xobs2)

    #xobs = np.concatenate((xobs1, xobs2))
    #yobs = np.concatenate((yobs1, yobs2))

    #dmx = max(xobs) - min(xobs)
    #dmy = max(yobs) - min(yobs)

    liten = 0.1  # 10 cm distance between wells
    var_intervall = maxdist / ant

    hegam = np.zeros((ant + 2, 12))

    hegam[:, 7] = np.max([zo1, zo2])

    nugget_distance = var_intervall - liten

    for t in range(ant + 2):
        hegam[t, 0] = t
        hegam[t, 1] = var_intervall * t - nugget_distance

    for i in range(N1):
        print(f'{100 * i / N1:.2f}%')

        for j in range(i, N2):
            dx = xobs1[i] - xobs2[j]
            dy = yobs1[i] - yobs2[j]

            h = np.sqrt(dx**2 + dy**2)

            if h < maxdist:
                intrin = (zo1[i] - zo2[j])**2

                t = 1
                while h > hegam[t + 1, 1]:
                    t += 1

                if t < ant + 2:
                    hegam[t, 2] += intrin
                    hegam[t, 3] += 1
                    hegam[t, 4] += zo1[i] + zo2[j]
                    hegam[t, 5] += zo1[i]**2 + zo2[j]**2
                    hegam[t, 6] += intrin**2

                    if hegam[t, 8] > intrin:
                        hegam[t, 8] = intrin
                    if hegam[t, 9] < intrin:
                        hegam[t, 9] = intrin

    hegam[:, 3] = hegam[:, 2] / hegam[:, 3]
    hegam[:, 5] = hegam[:, 4] / hegam[:, 3]
    hegam[:, 6] = hegam[:, 6] / (hegam[:, 3] - 1) - (hegam[:, 5]**2) * (hegam[:, 3] / (hegam[:, 3] - 1))
    hegam[:, 7] = (hegam[:, 7] - hegam[:, 3] * (hegam[:, 2]**2)) / hegam[:, 3]
    hegam[:, 7] = np.sqrt(hegam[:, 7])
    hegam[:, 2] = hegam[:, 2] / 2

    return hegam


# Example usage:
Z1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Z2 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
ant = 10
maxdist = 20

hegam_result = funk_semivar_mean_var(Z1, Z2, ant, maxdist)
print(hegam_result)
