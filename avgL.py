
gGain=1
rlastGain=1.4608
blastGain=1.4358
rcurGain=1.4986
bcurGain=1.4216
newAvgL=36.39



curAvgL=newAvgL*(0.299*rcurGain+0.587*gGain+0.114*bcurGain)/(0.299*rlastGain+0.587*gGain+0.114*blastGain)

print("curAvgL",curAvgL)

print(56/curAvgL)