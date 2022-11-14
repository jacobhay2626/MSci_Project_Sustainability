import pandas as pd

''' 
EXPANDING TO WHOLE DATA SET
    Need to create a function that allows us to search for a chemical
    Could use an input so the user inputs the function they want the score for
    We then retrieve those values for that chemical. 
    First need to know what number the row is for that chemical then use that to retrieve the values.   
    
'''

data = pd.read_csv(r'/Users/jacobhayward/Desktop/Methanol.csv')
df = pd.DataFrame(data, columns=['Solvent Name', 'Flash Point', 'Boiling point', 'Resistivity', 'Peroxide Formation',
                                 'Autoignition Temperature', 'Energy of Decomposition', 'H224/H225', 'H226'])

User_input = input("Chemical Name: ")
# Get the solvent they want to find out the score

Chemical_name = df.loc[df['Solvent Name'] == User_input].index
# find the row index for that solvent
# Giving the index in this form: Int64Index([0], dtype='int64')
# Need to convert that form to just a zero (EG)
Chemical_name_index = Chemical_name[0]

'''
retrieve the values for the different properties of the solvent. 
'''
Flash_Point = df.loc[Chemical_name_index].at['Flash Point']
GHS_224_225 = df.loc[Chemical_name_index].at['H224/H225']
GHS_226 = df.loc[Chemical_name_index].at['H226']
AIT = df.loc[Chemical_name_index].at['Autoignition Temperature']
RES = df.loc[Chemical_name_index].at['Resistivity']
PER = df.loc[Chemical_name_index].at['Peroxide Formation']
EOD = df.loc[Chemical_name_index].at['Energy of Decomposition']


def safety(fp, ghs1, ghs2, ait, res, perox, eod):
    # If flash point is within given parameters then return a score.
    score = 0
    # if the flash point is over 60 automatically assign a score of 0
    if fp > 60:
        score += 1

    # if the fp is below 60 enter the while loop

    while fp < 60:

        if fp >= 24:
            score += 3
        elif (fp >= 0) & (fp < 23):
            score += 4
        elif fp >= -20 & fp <= -1:
            score += 5
        else:
            score += 7
        break
    # while the H224/H225 codes are present assign a score of three only if the score is already below three
    while ghs1 == 1:

        if score == 1:
            score += 2
        break

    # while H226 is present and the score is < 5 assign a score of 5

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


df2 = pd.DataFrame(data,
                   columns=['Solvent Name', 'Boiling point', 'H341/H351/H361', 'H340/H350/H360', 'H304/H371/H376', 'H334', 'H370/H372',
                            'H302/H312/H332/H336/EUH070', 'H301/H311/H331', 'H300/H310/H330',
                            'H315/H317/H319/H335/EUH066', 'H318', 'H314'])

BP = df2.loc[Chemical_name_index].at['Boiling point']
CMR6 = df2.loc[Chemical_name_index].at['H341/H351/H361']
CMR9 = df2.loc[Chemical_name_index].at['H340/H350/H360']
STOT2 = df2.loc[Chemical_name_index].at['H304/H371/H376']
STOT4 = df2.loc[Chemical_name_index].at['H334']
STOT6 = df2.loc[Chemical_name_index].at['H370/H372']
AcT2 = df2.loc[Chemical_name_index].at['H302/H312/H332/H336/EUH070']
AcT6 = df2.loc[Chemical_name_index].at['H301/H311/H331']
AcT9 = df2.loc[Chemical_name_index].at['H300/H310/H330']
Irr2 = df2.loc[Chemical_name_index].at['H315/H317/H319/H335/EUH066']
Irr4 = df2.loc[Chemical_name_index].at['H318']
Irr7 = df2.loc[Chemical_name_index].at['H314']
'''
Want to say if a certain code is present give a certain score
However, if there is a situation where they have a code that scores 6 and 
a code that scores 9, we want the final output to give the highest score?
Could add the score for a given code to an empty array and then 
print the highest value within that array at the end?
'''


def health(bp, cmr6, cmr9, stot2, stot4, stot6, act2, act6, act9, irr2, irr4, irr7):
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

    h_score = int(max(score))

    if bp < 85:
        h_score += 1

    return h_score


df3 = pd.DataFrame(data, columns=['Solvent Name', 'Boiling point', 'No H4xx after fully registered', 'H412/H413',
                                  'H400,H410,H411',
                                  'EUH420 (ozone)', 'No, or partial REACh  registration'])

BP = df3.loc[Chemical_name_index].at['Boiling point']
GHS3 = df3.loc[Chemical_name_index].at['No H4xx after fully registered']
GHS5 = df3.loc[Chemical_name_index].at['H412/H413']
GHS7 = df3.loc[Chemical_name_index].at['H400,H410,H411']
OTH5 = df3.loc[Chemical_name_index].at['No, or partial REACh  registration']
OZO = df3.loc[Chemical_name_index].at['EUH420 (ozone)']


def environmental(bp, ghs3, ghs5, ghs7, oth5, ozo):
    score = []

    if bp < 50:
        score.append("7")

    if bp > 200:
        score.append("7")

    if 50 <= bp <= 69:
        score.append("5")

    if 140 <= bp <= 200:
        score.append("5")

    if 70 <= bp <= 139:
        score.append("3")

    if ghs3 == 1:
        score.append("3")

    if ghs5 == 1:
        score.append("5")

    if ghs7 == 1:
        score.append("7")

    if oth5 == 1:
        score.append("5")

    if ozo == 1:
        score.append("10")

    return max(score)


Safety_score = safety(Flash_Point, GHS_224_225, GHS_226, AIT, RES, PER, EOD)
Health_score = health(BP, CMR6, CMR9, STOT2, STOT4, STOT6, AcT2, AcT6, AcT9, Irr2, Irr4, Irr7)
Env_score = environmental(BP, GHS3, GHS5, GHS7, OTH5, OZO)
print("Safety Score: " + str(Safety_score))
print("Health Score: " + str(Health_score))
print("Environment Score: " + str(Env_score))
