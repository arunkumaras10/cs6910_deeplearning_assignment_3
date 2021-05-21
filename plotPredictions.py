# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:11:57 2021

@author: Arun-PC
"""
import matplotlib.pyplot as plt
# This code is contributed by gauravrajput1

# Python3 program to print one possible
# way of converting a string to another

# Function to print the steps
def printChanges(s1, s2, dp):
    
    changes = {}
    i = len(s1)
    j = len(s2)
    
# Check till the end
    while(i > 0 and j > 0):
        
        # If characters are same
        if s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
            
        # Replace
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            # print("change", s1[i - 1],
                    # "to", s2[j - 1])
            changes[i-1] = 'r'
            j -= 1
            i -= 1
            
        # Delete
        elif dp[i][j] == dp[i - 1][j] + 1:
            # print("Delete", s1[i - 1])
            changes[i-1] = 'd'
            i -= 1
            
        # Add
        elif dp[i][j] == dp[i][j - 1] + 1:
            # print("Add", s2[j - 1])
            changes[i] = 'a'
            j -= 1
    return changes
            
# Function to compute the DP matrix
def editDP(s1, s2):
    
    len1 = len(s1)
    len2 = len(s2)
    dp = [[0 for i in range(len2 + 1)]
            for j in range(len1 + 1)]
    
    # Initialize by the maximum edits possible
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # Compute the DP Matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            
            # If the characters are same
            # no changes required
            if s2[j - 1] == s1[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                
            # Minimum of three operations possible
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],
                                dp[i - 1][j - 1],
                                dp[i - 1][j])
                                    
    # Print the steps
    return printChanges(s1, s2, dp)

# Driver Code அசோக்,அசோகா
s1 = "அசோக்"
s2 = "அசோகா"
words = [('asok',s1,s2), ('pazhani','பழனி','பழநி'), ('pazhuvettaraiyar','பழுவீற்றையாளர்','பழுவேட்டரையர்'),
         ('baangaak','பாங்காக்','பாங்காக்'), ('pandavargal','பண்டவர்கள்','பாண்டவர்கள்'),
         ('muhaideen','முகைடென்','முகைதீன்'), ('muslim','முஸ்லீம்','முசுலீம்'),
         ('munmaadhiri','முன்மதிரி','முன்மாதிரி'), ('moonrarai','மோன்றரை','மூன்றரை'),
         ('raajaraajanin','இராஜராஜனின்','ராஜராஜனின்')]

half_matras = set([  'ா'    , 'ி'    , 'ீ'    , 'ு'    , 'ூ'    , 'ெ'    , 'ே'    , 'ை'    , 'ொ'    , 'ோ'    , 'ௌ'    , '்' ])
colorcode = {'a': 'rgba(0,255,0,0.4)', 'r' : 'rgba(0,0,255,0.2)', 'd' : 'rgba(255,0,0,0.2)'}
html = '<html><head><meta charset="UTF-8"><style>table, th, td {border: 1px solid black;border-collapse: collapse;}th {text-align: left;}th,td {padding: 5px;}</style></head>'
html += '<body><table>'
html += '<tr><th>Source</th><th>Ground Truth</th><th>Prediction</th><th>Edit Distance</th><th>Errors based on edit distance</th></tr>'
for source, pred, truth in words:
    changes = editDP(pred, truth)
    new_changes = {}
    for i, action in changes.items():
        if pred[i] in half_matras:
            new_changes[i-1] = action
        else:
            new_changes[i] = action
    changes = new_changes
    row = '<tr>'
    row += ('<td>%s</td>') % source
    row += ('<td>%s</td>') % truth
    row += ('<td>%s</td>') % pred
    row += ('<td>%d</td>') % len(changes)
    row += '<td>'
    for i,c in enumerate(pred):
        if i in changes:
            row += '<span style="display:inline; font-size: x-large; font-weight: bold; background-color:%s;">' % colorcode[changes[i]]
        else:
            row += '<span style="display:inline; font-size: x-large; font-weight: bold;">'
        if i in changes:
            if changes[i] == 'r' or changes[i] == 'd':
                row += (str(c) + '</span>')
            else:
                row += ('_</span>') + ('<span style="display:inline; font-size: x-large; font-weight: bold;">' + str(c) + '</span>')
        else:
            row += (str(c) + '</span>')
        
    row += '</td></tr>'
    html += row
html += '</table></body></html>'

with open('prediction_grid.html', 'w', encoding='utf-8') as f:
    f.write(html)
    