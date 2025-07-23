import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def roc_uplift_by_features(model, argList, x_train, y_train):
    
    argListCopy = argList.copy()
    featureList = []
    best_roc = 0
    best_roc2 = 0
    current_roc = 0
    
    df_res = pd.DataFrame(columns = ['Feature', 'Cummulative ROC', 'Uplift ROC Abs', 'Precision Left', 'Recall Right', 'Uplift ROC Rel2'])
    ind = 0
    
    while len(argListCopy) > 0:
        for feature in argListCopy:
            featureList.append(feature)
            model.fit(x_train[featureList], y_train)
            roc = roc_auc_score(y_train, model.predict_proba(x_train[featureList])[:, 1])
            featureList.remove(feature)
            
            if roc > best_roc: 
                best_feature = feature
                best_roc2 = best_roc
                best_roc = roc
            else:
                if roc > best_roc2:
                    best_roc2 = roc
         
        featureList.append(best_feature)
        argListCopy.remove(best_feature)
        
        
        model.fit(x_train[featureList], y_train)
        df_prob = pd.DataFrame({'Target': y_train, 'Probs': model.predict_proba(x_train[featureList])[:, 1]})
        df_prob.sort_values(by = 'Probs', inplace = True)
        df_prob.reset_index(drop = True, inplace = True)
        n = df_prob.shape[0]
        n_1 = df_prob['Target'].sum()
        n_0 = n - n_1
        rec = df_prob.iloc[n_0 : ]['Target'].sum() / n_1
        pr = (n_0 - df_prob.iloc[ : n_0 + 1]['Target'].sum()) / n_0
        
        
        
        df_res.loc[ind] = [best_feature, best_roc, best_roc - current_roc, pr, rec, (best_roc - best_roc2) / best_roc2]
        current_roc = best_roc
        best_roc = 0
        best_roc2 = 0
        ind += 1
    
    df_res['Uplift ROC Rel'] = df_res['Uplift ROC Abs'] / df_res.iloc[0]['Uplift ROC Abs']
    return df_res  



def plot_roc_uplift_by_features(df, rus_name):
    
    # Рисунок 1
    df_copy = df.copy()
    df_copy['Feature Rus'] = df_copy['Feature']

    for i in rus_name:
        df_copy['Feature Rus'][df_copy['Feature'] == i] = rus_name[i]
        
    n = df_copy.shape[0]
    fig, ax = plt.subplots(1, 1, figsize = (n, 0.7 * n), subplot_kw = dict(frame_on = False))
    
    ax.barh(
        y = df_copy['Feature Rus'].iloc[ : 0 : -1],
        width = df_copy['Uplift ROC Abs'].iloc[ : 0 : -1],
        color = "lightBlue")
    
    offset = df_copy['Uplift ROC Abs'].iloc[ : 0 : -1].max() * 1.025
    ax.set_xlim((0, offset))
    
    for i in range(n - 1):
        rel_uplift = df_copy.iloc[n - i - 1]['Uplift ROC Rel']
        if rel_uplift > 0:
            ax.text(offset * 1.05, i - 0.125, '+ ' + str(round(rel_uplift * 100, 2)) + '%', color = 'g')
        else:
            ax.text(offset * 1.05, i - 0.125, '- ' + str(round(rel_uplift * 100, 2)) + '%', color = 'r')
    
    ax.set_xlabel('Abs. Uplift'); 
    ax.set_title('Uplift ROC AUC относительно качества на самой сильной фиче \n Самая сильная фича: '\
              + df_copy['Feature Rus'].iloc[0] + ' (ROC AUC: ' + str(round(df_copy['Uplift ROC Abs'].iloc[0], 2)) + ')')
    
    plt.show()
    
    
    
    # Рисунок 2
    fig, ax = plt.subplots(1, 1, figsize = (n, 0.7 * n), subplot_kw = dict(frame_on = False))
    
    ax.barh(
        y = df_copy['Feature Rus'].iloc[-2 :  : -1],
        width = df_copy['Uplift ROC Rel2'].iloc[-2 :  : -1],
        color = "lightseagreen")
    
    offset = df_copy['Uplift ROC Rel2'].iloc[-2 :  : -1].max() * 1.025
    ax.set_xlim((0, offset))
    
    for i in range(n - 1):
        rel_uplift = df_copy.iloc[n - i - 2]['Uplift ROC Rel2']
        ax.text(offset * 1.05, i - 0.125, '+ ' + str(round(rel_uplift * 100, 2)) + '%', color = 'g')
    
    ax.set_xlabel('Rel. Uplift'); 
    ax.set_title('Uplift ROC AUC: сравнение двух наиболее сильных фичей на каждом шаге')
    
    plt.show()

    
   