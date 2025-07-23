import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns

def get_roc_auc_plot(X_train, y_train, X_test, y_test, X_oot, y_oot, featuresList, featuresDict, model, modelName, targetName):
    """
    Функция принимает на вход Train и Test выборки, а также ООТ выборку, список фичей, словарь для расшифровки фичей, обученную модель и таргет.
    Таргет принимает значения APPL/DEAL
    """
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')

    # TRAIN
    y_pred_train = model.predict_proba(X_train[featuresList])[:, 1]
    fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_pred_train)

    # TEST
    y_pred_test = model.predict_proba(X_test[featuresList])[:, 1]
    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_pred_test)

    # OOT
    y_pred_oot = model.predict_proba(X_oot[featuresList])[:, 1]
    fpr_oot, tpr_oot, threshold_oot = roc_curve(y_oot, y_pred_oot)
    roc_auc_oot = roc_auc_score(y_oot, y_pred_oot)

    fig, ax = plt.subplots(1, 1, figsize = (15, 15))
    ax.plot([0, 1], label='Baseline', linewidth=2)
    ax.plot(fpr_train, tpr_train, label = 'Train ({})'.format(round(roc_auc_train, 3)), color='red')
    ax.plot(fpr_test, tpr_test, label = 'Test ({})'.format(round(roc_auc_test, 3)), color='orange')
    ax.plot(fpr_oot, tpr_oot, label = 'OOT ({})'.format(round(roc_auc_oot, 3)), color='green')
    ax.legend();
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); 
    ax.set_title(targetName + ' ROC AUC ' + modelName)
    ax.patch.set_alpha(1.0)
    fig.patch.set_facecolor('#FFFFFF')
    fig.patch.set_alpha(0.0)
    # plt.savefig(f"{targetName}_metrics.png")
    plt.show()

    
    try:
        fig, ax = plt.subplots(1, 1, figsize = (15, 15))
        # if modelName=='LightGBM':
        #     feat_importance = pd.Series(model.booster_.feature_importance(importance_type='impurity'), index=[featuresDict[featureName] for featureName in X_train_appl.columns]).sort_values()
        # else:
        feat_importance = pd.Series(model.feature_importances_, index=[featuresDict[featureName] for featureName in X_train[featuresList].columns]).sort_values()
        feat_importance.plot(kind='barh', ax=ax)
        ax.set_title('Feature Importance')
        ax.patch.set_alpha(1.0)
        fig.patch.set_facecolor('#FFFFFF')
        fig.patch.set_alpha(0.0)
        # plt.savefig(f"{targetName}_feat_imp.png")
        plt.show()
    except:
        print('ГРАФИК ЗНАЧИМОСТИ ФИЧЕЙ НЕ ПОСТРОЕН')
        return 0