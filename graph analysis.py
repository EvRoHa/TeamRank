import numpy as np
from scipy.sparse import csc_matrix
from math import e

TEAMS = ["Air Force", "Akron", "Alabama", "Appalachian State", "Arizona", "Arizona State", "Arkansas",
         "Arkansas State", "Army", "Auburn", "BYU", "Ball State", "Baylor", "Boise State", "Boston College",
         "Bowling Green", "Buffalo", "California", "Central Michigan", "Charlotte", "Cincinnati", "Clemson",
         "Coastal Carolina", "Colorado", "Colorado State", "Connecticut", "Duke", "East Carolina", "Eastern Michigan",
         "Florida", "Florida Atlantic", "Florida International", "Florida State", "Fresno State", "Georgia",
         "Georgia Southern", "Georgia State", "Georgia Tech", "Hawai'i", "Houston", "Illinois", "Indiana", "Iowa",
         "Iowa State", "Kansas", "Kansas State", "Kent State", "Kentucky", "LSU", "Lafayette", "Liberty", "Louisiana",
         "Louisiana Monroe", "Louisiana Tech", "Louisville", "Marshall", "Maryland", "Memphis", "Miami", "Miami (OH)",
         "Michigan", "Michigan State", "Middle Tennessee", "Minnesota", "Mississippi State", "Missouri", "NC State",
         "Navy", "Nebraska", "Nevada", "New Mexico", "New Mexico State", "North Carolina", "North Texas",
         "Northern Illinois", "Northwestern", "Notre Dame", "Ohio", "Ohio State", "Oklahoma", "Oklahoma State",
         "Old Dominion", "Ole Miss", "Oregon", "Oregon State", "Penn State", "Pittsburgh", "Purdue", "Rice", "Rutgers",
         "SMU", "San Diego State", "San José State", "South Alabama", "South Carolina", "South Florida", "Stanford",
         "Syracuse", "TCU", "Temple", "Tennessee", "Texas", "Texas A&M", "Texas State", "Texas Tech", "Toledo", "Troy",
         "Tulane", "Tulsa", "UAB", "UCF", "UCLA", "UMass", "UNLV", "USC", "UT San Antonio", "UTEP", "Utah",
         "Utah State", "Vanderbilt", "Virginia", "Virginia Tech", "Wake Forest", "Washington", "Washington State",
         "West Virginia", "Western Michigan", "Wisconsin", "Wyoming"]

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

    return L / (1 + e**(-k*(x-x0)))+y0

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
            data[data < 0] = vfunc(data[data < 0], L=kwargs['L'], k=kwargs['k'], x0=kwargs['x0'])
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
    with open('matrix.txt', 'r') as infile:
        data = np.genfromtxt(infile, dtype=float, delimiter=',', skip_header=0, autostrip=True)
    result = TeamRank(data, MOV=True, transform=possessions)
    result = [x for _, x in sorted(zip(result, TEAMS), reverse=True)]
    for x in enumerate(result):
        print(x[0] + 1, x[1])
