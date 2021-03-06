import numpy as np
from scipy.sparse import csc_matrix
from math import e
import requests
import json

def possessions(x):
    td, fg, result = 0, 0, 0
    while x >= 8:
        td += 1
        result += 1
        x -= 8
    while x >= 3:
        fg += 1
        result += 1
        x -= 3
    return result


def logistic(x, L=1, k=1, x0=0, y0=0, y1=0):
    """
    :param x: the value to transform
    :param L: the curve's maximum value
    :param k: the logistic growth rate of the curve
    :param x0: the x-value of the curve's midpoint
    :param y0: the y-value of the curve's midpoint
    :return: float, the transformed value
    """
    # since the y-value of the midpoint is always halfway up the curve, we have to scale things a bit.
    y0 -= L
    L -= y0

    return L / (1 + e ** (-k * (x - x0))) + y0


def cap(x, xmax=28.):
    return max(x, xmax)


def TeamRank(data, p=0.85, MOV=True, transform=None, **kwargs):
    """
    :param data: an adjacency matrix
    :param p: the probability of a teleportation or sink
    :param MOV: a boolean about whether MOV should be considered
    :param transform: a function that takes an float and gives a float; used to transform the MOV
    :return:
    """
    # to create the transition matrix, ignore all of the wins; transform the losses into positively weighted outbound edges
    data[data > 0] = 0

    # if links are MOV weighted, convert all negative margins to represent positive loss margins. Otherwise, set to 1.
    if MOV:
        if transform == logistic:
            vfunc = np.vectorize(transform, excluded=['k', 'L', 'x0'])
            data[data < 0] = vfunc(data[data < 0])
        elif transform == cap:
            vfunc = np.vectorize(cap, excluded=['xmax'])
            data[data < 0] = vfunc(data[data < 0], xmax=kwargs['xmax'])
        elif transform == possessions:
            vfunc = np.vectorize(possessions)
            data[data < 0] *= -1
            data = vfunc(data)
        else:
            data[data < 0] *= -1
    else:
        data[data < 0] = 1

    A = csc_matrix(data, dtype=float)
    rsums = np.array(A.sum(0))[:, 0]
    A.data /= rsums

    # bool array  of sink states
    sink = rsums == 0

    s = data.shape[0]
    r0, r = np.zeros(s), np.ones(s)

    # set the change threshold
    maxerr = 0.00001

    # account for sink states
    D = sink / float(s)

    # account for teleportation out of undefeated teams
    E = np.ones(s) / float(s)

    trans = D * p + E * (1 - p)
    while True:
        r0 = r.copy()
        for i in range(0, s):
            # losses to team_i
            A_i = np.array(A[:, i].todense())[:, 0]
            r[i] = r0.dot(A_i * p + trans)
        err = np.sum(np.abs(r - r0))
        if err > maxerr:
            break

    return r / float(sum(r))


if __name__ == '__main__':
    # Build the request URL
    url = 'https://api.collegefootballdata.com/games?&year=2019'

    # Get the data
    data = json.loads(requests.get(url=url).text)

    # Get a sorted list of FBS teams
    teams = sorted(list(set([x['home_team'] for x in data])))

    data = np.genfromtxt('matrix.txt', dtype=float, delimiter=',', skip_header=0, autostrip=True)

    pos = sorted(zip(TeamRank(data.copy(), MOV=True, transform=possessions), teams), reverse=True)
    log = sorted(zip(TeamRank(data.copy(), MOV=True, transform=logistic), teams), reverse=True)
    points = sorted(zip(TeamRank(data.copy(), MOV=True), teams), reverse=True)
    noMOV = sorted(zip(TeamRank(data.copy(), MOV=False), teams), reverse=True)

    for x in range(len(pos)):
        print('{}|{}|{}|{}|{}'.format(x + 1, noMOV[x][1], points[x][1], pos[x][1], log[x][1]))
