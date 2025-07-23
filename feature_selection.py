'''
Эта библиотека содержит в себе методы отбора фичей:

1. backward_selection
    -- классический backward selection (исключение фичей по одной)
    -- backward selection до заданного числа фичей

2. forward_selection
    -- классический forward selection (добавление фичей по одной)
    -- forward selection до заданного числа фичей
    -- forward selection, добавляющий по m фичей, если улучшение качества при добавлении m - 1 фичи было ниже качества, задаваемого пользователем

3. bidirectional_selection
    В результате работы метод сходится к набору фичей размерности m / 2, где m -- размер начального набора фичей

4. SFFS (Sequential Floating Forward Selection)
    На каждой итерации метод добавляет одну лучшую фичу, затем удаляет худшие фичи из добавленных (должно быть улучшение качества выше определнного порога)

5. SFBS (Sequential Floating Backward Selection)
    На каждой итерации метод исключает одну самую незначащую фичу, затем пытается добавить лучшие фичи (должно быть улучшение качества выше определнного порога)
'''


from itertools import combinations
from sklearn.metrics import roc_auc_score
from termcolor import colored


def backward_selection(argList, model, x_train, y_train, x_test, y_test, quality_loss = 0.5, n_features_to_select = -1):
    """
    Входные данные:
        - argList               -- список фичей
        - model                 -- необученная модель
        - x_train               -- фичи для обучения модели (можно передать вместе с таргетом)
        - y_train               -- таргет для обучения
        - x_test                -- test датасет
        - y_test                -- таргет для test датасета
        - quality_loss          -- максимальный процент потери качества на backward шаге относительно исходного набора фичей. Например, 0.2%. По умолчанию 0.5%
        - n_features_to_select  -- точное число фичей, которое хотим получить на выходе. -1, если ограничения на число фичей нет
    """
    backward_argList = argList.copy()   

    # fit and predict на исходном наборе фичей. Метрика на исходном наборе фичей
    model.fit(x_train[backward_argList], y_train)
    probs_test = model.predict_proba(x_test[backward_argList])[:, 1]
    roc = roc_auc_score(y_test, probs_test)
    print('ROC AUC Test initial= ', roc)

    if n_features_to_select == -1:
        while True:
            best_roc_dif = 0
            
            # fit и predict на всех фичах, кроме i. Вычисление метрики
            for i in backward_argList: 
                model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
                probs_test = model.predict_proba(x_test[backward_argList].drop(i, axis = 1))[:, 1]
                roc_new = roc_auc_score(y_test, probs_test)

                # Если качество на тесте не просело
                if (roc_new / roc >= 1 - quality_loss / 100):
                    # Сравнение падений качества и выбор фичи, которая меньше всего влияет на падение качества
                    if (roc_new / roc > best_roc_dif):
                        best_roc_dif = roc_new / roc
                        perem_del = i

            # Если потери качества весомые, то не выкидываем никакие фичи и завершаем цикл
            if (best_roc_dif == 0):
                break
            # Иначе удалям фичу из рассмотрения
            else:
                backward_argList.remove(perem_del) 
                print(f"Осталось фичей: {len(backward_argList)}, ROC_AUC = {best_roc_dif * roc}, удалена фича {perem_del}" )
            
        return backward_argList
    else:
        while len(backward_argList) > n_features_to_select:
            best_roc = 0
            
            for i in backward_argList: 
                model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
                probs_test = model.predict_proba(x_test[backward_argList].drop(i, axis = 1))[:, 1]
                roc_new = roc_auc_score(y_test, probs_test)

                if roc_new > best_roc:
                    best_roc = roc_new
                    perem_del = i

            backward_argList.remove(perem_del) 
            # Это можно закомментировать, но нужно будет убрать вывод ROC_AUC ниже
            model.fit(x_train[backward_argList], y_train)
            probs_test = model.predict_proba(x_test[backward_argList])[:, 1]
            roc_new = roc_auc_score(y_test, probs_test)
            print(f"Осталось фичей: {len(backward_argList)}, ROC_AUC = {roc_new}, удалена фича {perem_del}" )
            
        return backward_argList


def forward_selection(argList, model, x_train, y_train, x_test, y_test, max_add_features_per_iter, quality_improve, max_features_to_select, n_features_to_select = -1):
    """
    Входные данные:
        - argList                   -- список фичей
        - model                     -- необученная модель
        - x_train                   -- фичи для обучения модели (можно передать вместе с таргетом)
        - y_train                   -- таргет для обучения
        - x_test                    -- test датасет
        - y_test                    -- таргет для test датасета
        - max_add_features_per_iter -- сколько максимум фичей можем добавить за один раз
        - quality_improve           -- список размерности k. Содержит в себе проценты улучшений, которые мы хотим видеть при добавлении 1, 2, ..., k фичей. Например, [1, 1.2, 1.5]
        - max_features_to_select    -- максимальное количество фичей, которое хотим увидеть в результате. Для значения 10 на выходе можем получить 10, 9, 8, 7 фичей. При указании положительно параметра n_features_to_select, игнорируется.
        - n_features_to_select      -- точное число фичей, которое хотим получить на выходе. -1, если ограничения на число фичей нет (если указано положительное число, то фичи будут добавляться по одной)

        combinations -- метод библиотеки itertools. from itertools import combinations
    """
    if max_features_to_select > len(argList):
        max_features_to_select = len(argList)

    if len(quality_improve) != max_add_features_per_iter:
        quality_improve = [0.4 * i for i in range(1, max_add_features_per_iter + 1)]

    forward_argList = [] 

    best_roc_auc = 1.e-2
    if n_features_to_select == -1:
        for k in range(1, max_add_features_per_iter + 1):

            print(10*'=' + f' Добавляем фичи по {k} ' + 10*'=')
            if (len(forward_argList) + k > max_features_to_select):
                break
            
            while True:
                best_dif = 0
                for features in list(combinations(set(argList) - set(forward_argList), k)):
                    model.fit(x_train[forward_argList + list(features)], y_train)
                    probs_test = model.predict_proba(x_test[forward_argList + list(features)])[:, 1]
                    roc_auc = roc_auc_score(y_test, probs_test)

                    if roc_auc / best_roc_auc >= 1.00 + quality_improve[k - 1] / 100:
                        if roc_auc / best_roc_auc > best_dif:
                            best_dif = roc_auc / best_roc_auc
                            features_to_add = list(features)

                if (best_dif == 0):
                    break
                else: 
                    best_roc_auc *= best_dif
                    forward_argList += features_to_add
                    print(f'Фичей в наборе: {len(forward_argList)}, ROC_AUC = {best_roc_auc}')

        return forward_argList
    else:
        print(10*'=' + f' Добавляем фичи поштучно до {n_features_to_select} в итоговом наборе ' + 10*'=')
        
        while len(forward_argList) < n_features_to_select:
            best_roc = 0
            for feature in list(set(argList) - set(forward_argList)):
                model.fit(x_train[forward_argList + [feature]], y_train)
                probs_test = model.predict_proba(x_test[forward_argList + [feature]])[:, 1]
                roc_auc = roc_auc_score(y_test, probs_test)

                if roc_auc > best_roc:
                    best_roc = roc_auc
                    feature_to_add = feature


            
            forward_argList.append(feature_to_add)
            print(f'Фичей в наборе: {len(forward_argList)}, ROC_AUC = {best_roc}')   

        return forward_argList


def bidirectional_selection(argList, model, x_train, y_train, x_test, y_test):
    """
    Входные данные:
        - argList                   -- список фичей
        - model                     -- необученная модель
        - x_train                   -- фичи для обучения модели (можно передать вместе с таргетом)
        - y_train                   -- таргет для обучения
        - x_test                    -- test датасет
        - y_test                    -- таргет для test датасета
    """

    forward_argList = [] 
    backward_argList = argList.copy()

    while len(backward_argList) != len(forward_argList):

        # Шаг forward
        best_roc_auc = 0
        feature_to_add = []
        for feature in list(combinations(set(backward_argList) - set(forward_argList), 1)):
            model.fit(x_train[forward_argList + list(feature)], y_train)
            probs_test = model.predict_proba(x_test[forward_argList + list(feature)])[:, 1]
            roc_auc = roc_auc_score(y_test, probs_test)

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                feature_to_add = list(feature)

        
        forward_argList += feature_to_add
        print(f"Фичей в forward: {len(forward_argList)}, ROC_AUC = {best_roc_auc}", end='\t')

        # Шаг backward
        best_roc_auc = 0
        feature_to_drop = []
        for i in list(set(backward_argList) - set(forward_argList)): 
            model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
            probs_test = model.predict_proba(x_test[backward_argList].drop(i, axis = 1))[:, 1]
            roc_auc = roc_auc_score(y_test, probs_test)

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                feature_to_drop = i

        backward_argList = list(set(backward_argList) - set([feature_to_drop]))

        print(f"Фичей в backward: {len(backward_argList)}, ROC_AUC = {best_roc_auc}")
    return backward_argList


def SFFS(argList, model, x_train, y_train, x_test, y_test, quality_improve = 0, quality_loss = 0):
    """
    Sequential Floating Forward Selection
    Входные данные:
        - argList                   -- список фичей
        - model                     -- необученная модель
        - x_train                   -- фичи для обучения модели
        - y_train                   -- таргет для обучения
        - x_test                    -- test датасет
        - y_test                    -- таргет для test датасета
        - quality_improve           -- минимальный процент улучшения метрики на forward шаге. Например, 1.0% или 0.4%. По умолчаюнию 0
        - quality_loss              -- максимальный процент потери качества на backward шаге. Например, 0.2%. По умолчанию 0
    """
    best_argList = []
    # fit and predict на исходном наборе фичей. Метрика на исходном наборе фичей
    model.fit(x_train[argList], y_train)
    probs_test = model.predict_proba(x_test[argList])[:, 1]
    roc_auc = roc_auc_score(y_test, probs_test)
    print('ROC AUC Test initial= ', roc_auc)
    best_roc_auc = 0.01
    deleted_features = []

    # Пока добавление фичи положительно влияет на метрику
    while True:
        best_roc_dif = 0
        # Ищем фичу, которая лучше всего увеличивает метрику
        for feature in set(argList) - set(best_argList) - set(deleted_features):
            model.fit(x_train[best_argList + [feature]], y_train)
            probs_test = model.predict_proba(x_test[best_argList + [feature]])[:, 1]
            roc_auc = roc_auc_score(y_test, probs_test)
            if roc_auc / best_roc_auc > (1.0 + quality_improve / 100) and roc_auc / best_roc_auc > best_roc_dif :
                best_roc_dif = roc_auc / best_roc_auc
                feature_to_add = feature
        # Если метрика не улучшилась, то завершаем поиск фичей
        if best_roc_dif == 0:
            break
        # Иначе добавляем фичу в набор
        else:
            best_argList.append(feature_to_add)
            best_roc_auc *= best_roc_dif
            print('Добавлена фича ' + colored(f'{feature_to_add}', 'green', attrs=['bold']) + f', ROC_AUC = {best_roc_auc}')
        
        if len(best_argList) > 2:
            print('\n' + 10*'=' + ' Начало фазы исключения фичей ' + 10*'=')
            deleted_features = []
            while True:
                best_roc_dif = 0
                for feature in set(best_argList) - set([feature_to_add]):
                    model.fit(x_train[best_argList].drop(feature, axis = 1), y_train)
                    probs_test = model.predict_proba(x_test[best_argList].drop(feature, axis = 1))[:, 1]
                    roc_auc = roc_auc_score(y_test, probs_test)
                    if roc_auc / best_roc_auc > (1.0 - quality_loss / 100) and roc_auc / best_roc_auc > best_roc_dif :
                        best_roc_dif = roc_auc / best_roc_auc
                        feature_to_remove = feature
                
                if best_roc_dif == 0:
                    break
                else:
                    best_argList.remove(feature_to_remove)
                    deleted_features.append(feature_to_remove)
                    best_roc_auc *= best_roc_dif
                    print('Удалена фича ' + colored(f'{feature_to_remove}', 'red', attrs=['bold']) + f', ROC_AUC = {best_roc_auc}')
                    
            print(10*'=' + ' Конец фазы исключения фичей ' + 10*'=' + '\n')

    return best_argList


def SFBS(argList, model, x_train, y_train, x_test, y_test, quality_improve = 0, quality_loss = 0):
    """
    Sequential Floating Backward Selection
    Входные данные:
        - argList                   -- список фичей
        - model                     -- необученная модель
        - x_train                   -- фичи для обучения модели
        - y_train                   -- таргет для обучения
        - x_test                    -- test датасет
        - y_test                    -- таргет для test датасета
        - quality_improve           -- минимальный процент улучшения метрики на forward шаге относительно формирующегося нового набора фичей. Например, 1.0% или 0.4%. По умолчаюнию 0
        - quality_loss              -- максимальный процент потери качества на backward шаге относительно исходного набора фичей. Например, 0.2%. По умолчанию 0
    """
    best_argList = argList.copy()
    model.fit(x_train[argList], y_train)
    probs_test = model.predict_proba(x_test[argList])[:, 1]
    roc = roc_auc_score(y_test, probs_test)
    print('ROC AUC Test initial= ', roc)
    best_roc_auc = 0.01
    added_features = []

    # Пока исключение фичи положительно влияет на метрику
    while True:
        best_roc_dif = 0
        # Ищем фичу, которая лучше всего увеличивает метрику
        for feature in set(best_argList) - set(added_features):
            model.fit(x_train[best_argList].drop(feature, axis = 1), y_train)
            probs_test = model.predict_proba(x_test[best_argList].drop(feature, axis = 1))[:, 1]
            roc_auc = roc_auc_score(y_test, probs_test)
            if roc_auc / roc > (1.0 - quality_loss / 100) and roc_auc / roc > best_roc_dif :
                best_roc_dif = roc_auc / roc
                feature_to_remove = feature
        # Если упала слишком сильно, то завершаем поиск фичей
        if best_roc_dif == 0:
            break
        # Иначе исключаем фичу из набора
        else:
            best_argList.remove(feature_to_remove)
            best_roc_auc = best_roc_dif * roc
            print('Удалена фича ' + colored(f'{feature_to_remove}', 'red', attrs=['bold']) + f', ROC_AUC = {best_roc_auc}')
        
        if len(argList) - len(best_argList) > 2:
            print('\n' + 10*'=' + ' Начало фазы добавления фичей ' + 10*'=')
            while True:
                best_roc_dif = 0
                for feature in set(argList) - set(best_argList) - set([feature_to_remove]):
                    model.fit(x_train[best_argList + [feature]], y_train)
                    probs_test = model.predict_proba(x_test[best_argList + [feature]])[:, 1]
                    roc_auc = roc_auc_score(y_test, probs_test)
                    if roc_auc / best_roc_auc > (1.0 + quality_improve / 100) and roc_auc / best_roc_auc > best_roc_dif :
                        best_roc_dif = roc_auc / best_roc_auc
                        feature_to_add = feature
                
                if best_roc_dif == 0:
                    break
                else:
                    best_argList.append(feature_to_add)
                    added_features.append(feature_to_add)
                    best_roc_auc *= best_roc_dif 
                    print('Добавлена фича ' + colored(f'{feature_to_add}', 'green', attrs=['bold']))
                    
            print(10*'=' + ' Конец фазы добавления фичей ' + 10*'=' + '\n')

    return best_argList

