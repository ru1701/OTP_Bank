import math


def psi_index_calc(x_train, x_test, name):
    
    psi = 0
    total_cnt_tr = x_train.shape[0]
    total_cnt_test = x_test.shape[0]
    
    boundDownList = []
    boundUpList = []
    
    for i in range(20):
        if i == 0:
            boundDownList.append(x_train.sort_values(by = name)\
                         .reset_index()\
                         .drop('index', axis = 1)\
                         .iloc[int(i * 0.05 * total_cnt_tr)][0])
            
            boundUpList.append(x_train.sort_values(by = name)\
                                      .reset_index()\
                                      .drop('index', axis = 1)\
                                      .iloc[int((i + 1) * 0.05 * total_cnt_tr - 1)][0])
        else:
            if x_train.sort_values(by = name)\
                      .reset_index().drop('index', axis = 1)\
                      .iloc[int(i * 0.05 * total_cnt_tr)][0] == boundUpList[-1]:
                continue
            else:
                boundDownList.append(x_train.sort_values(by = name)\
                                            .reset_index()\
                                            .drop('index', axis = 1)\
                                            .iloc[int(i * 0.05 * total_cnt_tr)][0])
                
                boundUpList.append(x_train.sort_values(by = name)\
                                          .reset_index()\
                                          .drop('index', axis = 1)\
                                          .iloc[int((i + 1) * 0.05 * total_cnt_tr - 1)][0])
            
    
    boundDownList = [0] + boundDownList[1 : ]
    boundUpList = boundDownList[1 : ] + [1]
    
    i = 0
    for bound_down, bound_up in zip(boundDownList, boundUpList):
        if i == 0:
            share_tr = x_train[x_train[name] < bound_up].shape[0] / total_cnt_tr
            share_test = x_test[x_test[name] < bound_up].shape[0] / total_cnt_test   
        elif i == len(boundDownList) - 1:
            share_tr = x_train[x_train[name] >= bound_down].shape[0] / total_cnt_tr
            share_test = x_test[x_test[name] >= bound_down].shape[0] / total_cnt_test 
        else:
            share_tr = x_train[(x_train[name] >= bound_down) & (x_train[name] < bound_up)].shape[0] / total_cnt_tr
            share_test = x_test[(x_test[name] >= bound_down) & (x_test[name] < bound_up)].shape[0] / total_cnt_test 
    
        psi += (share_tr - share_test) * math.log(share_tr / share_test)
        i += 1
        print('Bound Down = ' + str(bound_down) + '  Bound Up = ' + str(bound_up) + '  Share(Train) = ' + str(share_tr))
 
    return psi