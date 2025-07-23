import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree



def get_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        classes = path[-1][0][0]
        l = np.argmax(classes)
        rules += [rule]
        
    return rules


def describe_target_rate(x, y, arg_name, target_name, max_depth = 2, min_samples_split = 0.1, min_samples_leaf = 0.1):
    dt = DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, random_state = 123)
    dt.fit(x[[arg_name]], y[target_name])
    
    rules = get_rules(dt, [arg_name])
    
    index = []
    target_cnt = []
    total_cnt = []
    rt = []
    sortList = []

    for i in rules:
        ind1 = i.find(arg_name)
        ind2 = i.find(')')
        rules1 = i[ind1 : ind2]

        ind3 = i[ind2 + 1 : ].find(arg_name) + ind2 + 1
        ind4 = i[ind2 + 1 : ].find(')') + ind2 + 1
        rules2 = i[ind3 : ind4]

        if rules1.find('<') > -1:
            if rules2.find('<') > -1:
                right = x[arg_name][x[arg_name] <= float(rules2[rules2.find('<=') + 2 :])].max()
                idx = x[[arg_name]][x[arg_name] <= right].index

                total_cnt.append(x[[arg_name]][x[arg_name] <= right].shape[0])
                target_cnt.append(y.loc[idx].sum()[0])
                index.append('<=' + str(round(right, 3)))
                rt.append(target_cnt[-1] / total_cnt[-1])
                sortList.append(right)
            elif rules2.find('>') > -1:
                left = x[arg_name][x[arg_name] >= float(rules2[rules2.find('>') + 2 :])].min()
                right = x[arg_name][x[arg_name] <= float(rules1[rules1.find('<=') + 2 :])].max() 
        
                if (left > float(rules2[rules2.find('>') + 2 :])):
                    idx = x[[arg_name]][(x[arg_name] <= right) & (x[arg_name] >= left)].index
                    total_cnt.append(x[[arg_name]][(x[arg_name] <= right) & (x[arg_name] >= left)].shape[0])
                    index.append('[' + str(round(left, 3)) + ', ' + str(round(right, 3)) + ']')
                else:
                    idx = x[[arg_name]][(x[arg_name] <= right) & (x[arg_name] > left)].index
                    total_cnt.append(x[[arg_name]][(x[arg_name] <= right) & (x[arg_name] > left)].shape[0])
                    index.append('(' + str(round(left, 3)) + ', ' + str(round(right, 3)) + ']')

                target_cnt.append(y.loc[idx].sum()[0])
                rt.append(target_cnt[-1] / total_cnt[-1])    
                sortList.append(left)
            else:
                right = x[arg_name][x[arg_name] <= float(rules1[rules1.find('<') + 2 :])].max()
                idx = x[[arg_name]][x[arg_name] <= right].index
                
                target_cnt.append(y.loc[idx].sum()[0])
                total_cnt.append(x[[arg_name]][x[arg_name] <= right].shape[0])
                index.append('<=' + str(round(right, 3)))
                rt.append(target_cnt[-1] / total_cnt[-1])
                sortList.append(right)
        else:
            if rules2.find('<') > -1:
                left = x[arg_name][x[arg_name] >= float(rules1[rules1.find('>') + 2 :])].min()
                right = x[arg_name][x[arg_name] <= float(rules2[rules2.find('<=') + 2 :]) ].max() 
                
                if (left > float(rules1[rules1.find('>') + 2 :])):
                    idx = x[[arg_name]][(x[arg_name] <= right) & (x[arg_name] >= left)].index
                    total_cnt.append(x[[arg_name]][(x[arg_name] <= right) & (x[arg_name] >= left)].shape[0])
                    index.append('[' + str(round(left, 3)) + ', ' + str(round(right, 3)) + ']')
                else:
                    idx = x[[arg_name]][(x[arg_name] <= right) & (x[arg_name] > left)].index
                    total_cnt.append(x[[arg_name]][(x[arg_name] <= right) & (x[arg_name] > left)].shape[0])
                    index.append('(' + str(round(left, 3)) + ', ' + str(round(right, 3)) + ']')               

                target_cnt.append(y.loc[idx].sum()[0])
                rt.append(target_cnt[-1] / total_cnt[-1]) 
                sortList.append(left)
            elif rules2.find('>') > -1:
                left = x[arg_name][x[arg_name] >= float(rules2[rules2.find('>') + 2 :])].min()

                if (left > float(rules2[rules2.find('>') + 2 :])):
                    idx = x[[arg_name]][x[arg_name] >= left].index
                    total_cnt.append(x[[arg_name]][x[arg_name] >= left].shape[0])
                    index.append('>=' + str(round(left, 3)))
                else:
                    idx = x[[arg_name]][x[arg_name] > left].index
                    total_cnt.append(x[[arg_name]][x[arg_name] > left].shape[0])
                    index.append('>' + str(round(left, 3)))

                target_cnt.append(y.loc[idx].sum()[0])
                rt.append(target_cnt[-1] / total_cnt[-1])
                sortList.append(left)
            else:
                left = x[arg_name][x[arg_name] >= float(rules1[rules1.find('>') + 2 :])].min()
                
                if (left > float(rules1[rules1.find('>') + 2 :])):
                    idx = x[[arg_name]][x[arg_name] >= left].index
                    total_cnt.append(x[[arg_name]][x[arg_name] >= left].shape[0])
                    index.append('>=' + str(round(left, 3)))
                else:
                    idx = x[[arg_name]][x[arg_name] > left].index
                    total_cnt.append(x[[arg_name]][x[arg_name] > left].shape[0])
                    index.append('>' + str(round(left, 3)))                
                
                target_cnt.append(y.loc[idx].sum()[0])
                rt.append(target_cnt[-1] / total_cnt[-1])
                sortList.append(left)

    return pd.DataFrame({'sum': target_cnt, 'size': total_cnt, 'rt': rt, 'sort': sortList}, index = index)\
             .sort_values(by = 'sort')\
             .drop('sort', axis = 1)

# Отрисовка

# Входные данные:
# - r   -- результат работы `describe_target_rate`
# - sz  -- длинна исходного вектора xs. если не указано, считаем, что из xs записи не выкидывали.

# Выходные данные:
# - 2 графика:
#   - левый  -- доля таргета в бакете
#   - правый -- количество в бакете от общего числа записей

def render_target_rate(r, feature_name, sz = None):
    primary_color = '#C1FD0D' #'#006649'
    secondary_color = '#FE7C32' # '#86A03A'
    
    target = 'rt'
    
    (fig, ax) = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 3))
    r[target].plot.barh(ax = ax[0], width = 0.75, color = primary_color)
    
    s = 0
    for i in range(r.shape[0]):
        if s < 0.0066 * len(str(r.index[i])):
            s = 0.0066 * len(str(r.index[i]))
        
    for i in range(r.shape[0]):
        ax[0].text(0, i - 0.24, f" {r[target].iloc[i]*100:2.01f}%", color = 'black', fontsize = '11')

        if r.shape[0] == 2:
            if i == 0:
                plt.figtext(-s, 0.3, f"{r.index[i]}", color = 'black', fontsize = '11')     
            else:
                plt.figtext(-s, 0.7, f"{r.index[i]}", color = 'black', fontsize = '11') 
                
        if r.shape[0] == 3:
            if i == 0:
                plt.figtext(-s, 0.27, f"{r.index[i]}", color = 'black', fontsize = '11') 
            elif i == 1:
                plt.figtext(-s, 0.52, f"{r.index[i]}", color = 'black', fontsize = '11')  
            else:
                plt.figtext(-s, 0.77, f"{r.index[i]}", color = 'black', fontsize = '11')  

        if r.shape[0] == 4:
            if i == 0:
                plt.figtext(-s, 0.25, f"{r.index[i]}", color = 'black', fontsize = '11') 
            elif i == 1:
                plt.figtext(-s, 0.43, f"{r.index[i]}", color = 'black', fontsize = '11') 
            elif i == 2:
                plt.figtext(-s, 0.61, f"{r.index[i]}", color = 'black', fontsize = '11') 
            else:
                plt.figtext(-s, 0.79, f"{r.index[i]}", color = 'black', fontsize = '11')
        
                  
                
    ax[0].xaxis.set_ticks([])
    ax[0].yaxis.set_ticks([])
    ax[0].yaxis.set_label_text('')
    ax[0].set_title('Доля таргета', fontsize = 12)


    r['size'].plot.barh(ax = ax[1], width = 0.75, tick_label = None, color = secondary_color)

    for i in range(r.shape[0]):
        c = r['size'].iloc[i]
        is_inside = c > r['size'].max()*0.1
        ax[1].text(
            0 if is_inside else c,
            i - 0.24,
            f" {c/sz*100:2.01f}%",
            color = 'black' if is_inside else 'k',
            fontsize = '11')
    ax[1].yaxis.set_ticks([])
    ax[1].yaxis.set_label_text('')
    ax[1].set_title('Доля выборки', fontsize = 12)

    fig.tight_layout()
    fig.patch.set_alpha(0)
    fig.suptitle(feature_name, y = 1.15, fontsize = 15)
    
    display(fig)
    plt.close(fig)






