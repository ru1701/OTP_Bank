import math
from scipy.stats import norm


def events_rate(x, y):
    return (y + 0.0) / x 

def calc_nonevent(x):
    return x[0] - x[1]

def get_bucket(df, target_name, var_name, ascending_f, n_threshold, p_threshold):
    
    #Для каждого значения считаем кол-во записей и Event'ов
    df = df.groupby(var_name).agg({target_name : ['count', 'sum', 'std']})
    df.columns = df.columns.droplevel(level = 0)
    df['events_rate'] = df.apply(lambda x: events_rate(x['count'], x['sum']), axis =  1)
    df = df.reset_index()
    
    df[var_name + '_min'] = df[var_name]
    df[var_name + '_max'] = df[var_name]
    
    # После сортировки, сначала должны идти значения с наибольшей долей Events (!!!по логике!!!)
    df.sort_values(by = [var_name], ascending = ascending_f, inplace = True)
    df = df.reset_index(drop = True)
    
    #Merge бакетов по Events Rate
    while True:
        break_f = True
        
        cnt = df.shape[0]
        if cnt <= 1:
            break   
            
        cur_i = 0
        next_i = 1
        while True:
            if (cnt <= 1) | (cur_i == (cnt - 1)):
                break

            cur_rate = df['events_rate'].loc[cur_i]
            next_rate = df['events_rate'].loc[next_i]

            if (cur_rate > next_rate):
                cur_i = next_i
                next_i = cur_i + 1
                continue
            else:
                break_f = False
                
                if (cur_i == 0):
                    j = next_i
                else:
                    prev_rate = df['events_rate'].loc[cur_i - 1]
                    j = next_i
                        
                count2 = df['count'].loc[cur_i] + df['count'].loc[j]
                sum2 = df['sum'].loc[cur_i] + df['sum'].loc[j]
                events_rate2 = (sum2 + 1.0) / count2
                std2 = math.sqrt((sum2 * (1 - 2 * events_rate2) + count2 * events_rate2**2) / (count2 - 1))

                df['count'].loc[j] = count2
                df['sum'].loc[j] = sum2                
                df['std'].loc[j] = std2
                df['events_rate'].loc[j] = events_rate2
                
                df[var_name + '_min'].loc[j] = min(df[var_name + '_min'].loc[cur_i],                                                          df[var_name + '_min'].loc[j])
                df[var_name + '_max'].loc[j] = max(df[var_name + '_max'].loc[cur_i],                                                          df[var_name + '_max'].loc[j])

                df.drop([cur_i], inplace = True)
                df = df.reset_index(drop = True)
                cnt = df.shape[0]
                
                if (cur_i > 0):
                    cur_i = cur_i - 1
                    next_i = next_i - 1
                    
                continue
                
        if (break_f == True):
            break
    
    #Merge бакетов по P-Value
    while True:
        if cnt <= 1:
            break
            
        pval_list = []
        count_list = []
        event_list = []
        std_list = []
        
        for i in df.index:
            if i == 0:
                continue
                
            count_list.append(df['count'].loc[i] + df['count'].loc[i - 1])
            event_list.append(df['sum'].loc[i] + df['sum'].loc[i - 1])
            
            events_rate2 = (event_list[i - 1] + 1.0) / count_list[i - 1]
            std_list.append((event_list[i - 1] * (1 - 2 * events_rate2) + 
                             count_list[i - 1] * events_rate2**2) / (count_list[i - 1] - 2))
            
            if (std_list[i - 1] > 0):
                zval = (df['events_rate'].loc[i - 1] - df['events_rate'].loc[i] + 0.0) /                        math.sqrt(std_list[i - 1] * (1.0 / df['count'].loc[i - 1] + 1.0 / df['count'].loc[i]))
                pval = 1 - norm.cdf(zval)
            else:
                pval = 2
                
            if (df['count'].loc[i - 1] < n_threshold) | (df['count'].loc[i] < n_threshold):
                pval = pval + 1
                
            pval_list.append(pval)
            
        if(max(pval_list) < p_threshold):
            break
        else:
            ind_max = pval_list.index(max(pval_list))
            
            df['count'].loc[ind_max] = count_list[ind_max]
            df['sum'].loc[ind_max] = event_list[ind_max]                
            df['std'].loc[ind_max] = std_list[ind_max] * (count_list[ind_max] - 2) / (count_list[ind_max] - 1) 
            df['events_rate'].loc[ind_max] = (event_list[ind_max] + 1.0) / count_list[ind_max]
            
            df[var_name + '_min'].loc[ind_max] = min(df[var_name + '_min'].loc[ind_max],                                                          df[var_name + '_min'].loc[ind_max + 1])
            df[var_name + '_max'].loc[ind_max] = max(df[var_name + '_max'].loc[ind_max],                                                      df[var_name + '_max'].loc[ind_max + 1])

            df.drop([ind_max + 1], inplace = True)
            df = df.reset_index(drop = True)
            cnt = df.shape[0]
    
    df.rename(columns = {var_name + '_min': 'MIN_VALUE', var_name + '_max': 'MAX_VALUE'}, inplace = True)
        
    df = df.reset_index(drop = True)
    
    return df[['MIN_VALUE', 'MAX_VALUE', 'events_rate', 'count']]

