import sys
import numpy as np

def avg(filename):
    with open(filename,'r') as f:
        a = f.readline()
        a = a.strip('][')
        b = a.split()
        #print(b)
        tempsum = 0.0
        for num in b:
            tempsum = tempsum + float(num)
        tempsum = tempsum/len(b)
        return tempsum

def parse(fn):
    filename = fn[0:-3]
    tempstr = ''
    delimset = ['l','a','t','n']
    endSet = []
    for char in filename:
        if char in delimset:
            endSet.append(float(tempstr))
            tempstr = ''
        else :
            tempstr = tempstr + char
    endSet.append(avg(fn))
    return endSet

#keystrArr = ['layers','nodes','alpha','tolerance','initial_learning_rate','cross_val_score']

def aggregate(infile, outfile):
    out = []
    with open(infile,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            print(line)
            out.append(parse(line))
    #return out 
    #with open(outfile,'w') as g:
    #    import numpy as np
    #    a = np.array(out)
    #endArr = np.array(out)
    header='layers,nodes,alpha,tolerance,initial_learning_rate,cross_val_score'
    np.savetxt(outfile,np.array(out),delimiter=',',header=header)




if __name__ == '__main__':
  aggregate(sys.argv[1],sys.argv[2])
