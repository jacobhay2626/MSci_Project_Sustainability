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


df2 = pd.DataFrame(data, columns=['H341/H351/H361', 'H340/H350/H360', 'H304/H371/H376', 'H334', 'H370/H372',
                                  'H302/H312/H332/H336/EUH070', 'H301/H311/H331', 'H300/H310/H330',
                                  'H315/H317/H319/H335/EUH066', 'H318', 'H314'])

CMR6 = df2.loc[0].at['H341/H351/H361']
CMR9 = df2.loc[0].at['H340/H350/H360']
STOT2 = df2.loc[0].at['H304/H371/H376']
STOT4 = df2.loc[0].at['H334']
STOT6 = df2.loc[0].at['H370/H372']
AcT2 = df2.loc[0].at['H302/H312/H332/H336/EUH070']
AcT6 = df2.loc[0].at['H301/H311/H331']
AcT9 = df2.loc[0].at['H300/H310/H330']
Irr2 = df2.loc[0].at['H315/H317/H319/H335/EUH066']
Irr4 = df2.loc[0].at['H318']
Irr7 = df2.loc[0].at['H314']
'''
Want to say if a certain code is present give a certain score
However, if there is a situation where they have a code that scores 6 and 
a code that scores 9, we want the final output to give the highest score?
Could add the score for a given code to an empty array and then 
print the highest value within that array at the end?
'''


def health(cmr6, cmr9, stot2, stot4, stot6, act2, act6, act9, irr2, irr4, irr7):
    score = []

    if cmr6 == 1:
        score.append("6")

    if cmr9 == 1:
        score.append("9")

    if stot2 == 1:
        score.append("2")

    if stot4 == 1:
        score.append("4")

    if stot6 == 1:
        score.append("6")

    if act2 == 1:
        score.append("2")

    if act6 == 1:
        score.append("6")

    if act9 == 1:
        score.append("9")

    if irr2 == 1:
        score.append("2")

    if irr4 == 1:
        score.append("4")

    if irr7 == 1:
        score.append("7")

    return max(score)


Safety_score = safety(Flash_Point, GHS_224_225, GHS_226, AIT, RES, PER, EOD)
Health_score = health(CMR6, CMR9, STOT2, STOT4, STOT6, AcT2, AcT6, AcT9, Irr2, Irr4, Irr7)
print(Safety_score)
print(Health_score)
