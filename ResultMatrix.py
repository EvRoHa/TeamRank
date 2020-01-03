import requests
import json
import csv


def main(year):
    # Build the request URL
    url = 'https://api.collegefootballdata.com/games?&year={}'.format(year)

    # Get the data
    data = json.loads(requests.get(url=url).text)

    # Get a sorted list of FBS teams
    teams = sorted(list(set([x['home_team'] for x in data])))

    # Build an adjacency matrix
    adj_matrix = [[0 for x in teams] for y in teams]

    # Parse the json data to populate the adjacency matrix. 1 for a (home) win, 0 for no game
    # The row is the home team, the column is the away team.
    for x in data:
        try:
            h, a = teams.index(x['home_team']), teams.index(x['away_team'])

            spread = x['home_points'] - x['away_points']

            adj_matrix[h][a], adj_matrix[a][h] = spread, -spread

        except (ValueError, TypeError, KeyError):
            continue

    with open('matrix.txt', 'w+', newline='') as outfile:
        csvr = csv.writer(outfile)
        csvr.writerows(adj_matrix)

if __name__ == "__main__":
    main(2019)
