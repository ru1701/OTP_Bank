import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def stability_features(model, argList, x_train, y_train, type_test = 'shuffle', k = 20):
    
    model.fit(x_train[argList], y_train)
    cur_roc = roc_auc_score(y_train, model.predict_proba(x_train[argList])[:, 1])

    df_res = pd.DataFrame(columns = ['Feature', 'Stability Rel'])
    x_train_copy = x_train[:]
    
    for ind, name in enumerate(argList):
        roc = 0
        for i in range(k):
            
            if type_test == 'noise':
                x_train_copy[name + '_noise'] = x_train_copy[name] \
                      + np.random.normal(loc = 0, scale = x_train_copy[name].std(), size = x_train_copy.shape[0])   
                
                model.fit(x_train_copy[argList + [name + '_noise']].drop(name, axis = 1), y_train)       
                roc += roc_auc_score(y_train, model.predict_proba(x_train_copy[argList + [name + '_noise']].drop(name, axis = 1))[:, 1])
                
            elif type_test == 'shuffle':
                x_train_copy[name + '_shuffle'] = shuffle(x_train_copy[name]).to_numpy()
                model.fit(x_train_copy[argList + [name + '_shuffle']].drop(name, axis = 1), y_train)       
                roc += roc_auc_score(y_train, model.predict_proba(x_train_copy[argList + [name + '_shuffle']].drop(name, axis = 1))[:, 1]) 
                
            elif type_test == 'mixed':
                if k % 2 == 0:
                    x_train_copy[name + '_mixed'] = x_train_copy[name] \
                          + np.random.normal(loc = 0, scale = x_train_copy[name].std(), size = x_train_copy.shape[0])
                else:
                    x_train_copy[name + '_mixed'] = shuffle(x_train_copy[name]).to_numpy()
                    
                model.fit(x_train_copy[argList + [name + '_mixed']].drop(name, axis = 1), y_train)       
                roc += roc_auc_score(y_train, model.predict_proba(x_train_copy[argList + [name + '_mixed']].drop(name, axis = 1))[:, 1])
                
        roc = roc / k
        df_res.loc[ind] = [name, (cur_roc - roc) / cur_roc]
        
    return df_res 

def plot_feature_stability(df, rus_name):
    
    # Рисунок 1
    df_copy = df.copy()
    df_copy['Feature Rus'] = df_copy['Feature']
    df_copy['Stability Rel2'] = round(df_copy['Stability Rel'] * 100, 2) * (df_copy['Stability Rel'] > 0)
    df_copy['Stability Rel3'] = round(df_copy['Stability Rel'] * 100, 2) * (df_copy['Stability Rel'] < 0)

    for i in rus_name:
        df_copy['Feature Rus'][df_copy['Feature'] == i] = rus_name[i]
        
    n = df_copy.shape[0]
    fig, ax = plt.subplots(1, 1, figsize = (n + 1, 0.7 * (n + 1)), subplot_kw = dict(frame_on = False))
    
    h = 0.8
    
    ax.barh(
        y = df_copy['Feature Rus'],
        width = df_copy['Stability Rel2'],
        color = "darkseagreen")
    
    ax.barh(
        y = df_copy['Feature Rus'],
        width = df_copy['Stability Rel3'],
        left = df_copy['Stability Rel2'],
        height = h,
        color = "lightcoral")
    
    offset_max = df_copy['Stability Rel2'].max() * 1.025
    offset_min = df_copy['Stability Rel3'].min() * 1.025
    ax.set_xlim((offset_min, offset_max))
    
    for i in range(n):
        if df_copy.iloc[i]['Stability Rel2'] > 0:
            rel_stab = df_copy.iloc[i]['Stability Rel2']
        else:
            rel_stab = df_copy.iloc[i]['Stability Rel3']
            
        if rel_stab > 0:
            ax.text(offset_max * 1.05, i - 0.125, '+ ' + str(round(rel_stab, 2)) + '%', color = 'g')
        else:
            ax.text(offset_max * 1.05, i - 0.125, ' ' + str(round(rel_stab, 2)) + '%', color = 'r')
    
    ax.set_xlabel('%'); 
    ax.set_title('Стабильность модели при изменении значений признаков\n Изменение ROC AUC')
    
    plt.show()