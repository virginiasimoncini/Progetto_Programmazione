def knn(n,x,y,sample):
    #we save distances in a dictionary<index,distance>
    dist = dict()
    #calculate distance
    for index, row in df.iterrows():
        if row != sample:
            d = 0
            for column in df:
                d += (sample[column] - row[column])**2
            dict.update(index,Math.sqrt(d))
    return dist