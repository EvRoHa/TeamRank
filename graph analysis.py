import numpy as np
from scipy.sparse import csc_matrix

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


def TeamRank(data, p=0.85, MOV=True):
    # to create the transition matrix, ignore all of the wins; transform the losses into positively weighted outbound edges
    data[data > 0] = 0
    if MOV:
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
    result = TeamRank(data, MOV=False)
    result = [x for _, x in sorted(zip(result, TEAMS), reverse=True)]
    for x in enumerate(result):
        print(x[0] + 1, x[1])
