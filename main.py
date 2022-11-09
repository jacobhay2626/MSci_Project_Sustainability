import pandas as pd

data = pd.read_csv(r'/Users/jacobhayward/Desktop/Methanol.csv')
df = pd.DataFrame(data, columns=['Flash Point', 'Boiling point', 'Resistivity', 'Peroxide Formation',
                                 'Autoignition Temperature', 'Energy of Decomposition', 'H224/H225', 'H226'])

# 'Boiling point', 'Resistivity', 'Peroxide Formation',
#                                  'Autoignition Temperature', 'Energy of Decomposition', 'H224/H225', 'H226'

Flash_Point = df.loc[0].at['Flash Point']
GHS_224_225 = df.loc[0].at['H224/H225']
GHS_226 = df.loc[0].at['H226']
AIT = df.loc[0].at['Autoignition Temperature']
RES = df.loc[0].at['Resistivity']
PER = df.loc[0].at['Peroxide Formation']
EOD = df.loc[0].at['Energy of Decomposition']


def safety(fp, ghs1, ghs2, ait, res, perox, eod):
    # If flash point is within given parameters then return a score.
    score = 0

    if fp > 60:
        score += 1

    print(score)

    while fp < 60:

        if fp >= 24:
            score += 3
        elif fp >= 0 & fp < 23:
            score += 4
        elif fp >= -20 & fp <= -1:
            score += 5
        else:
            score += 7
        break

    while ghs1 == 1:

        if score == 1:
            score += 2
        break

    while ghs2 == 1:

        if score < 5:
            score = 5
        break

    if ait < 200:
        score += 1

    if res > 10000000000:
        score += 1

    if perox == "Yes":
        score += 1

    if eod > 500:
        score += 1

    return score


Safety_score = safety(Flash_Point, GHS_224_225, GHS_226, AIT, RES, PER, EOD)
print(Safety_score)
