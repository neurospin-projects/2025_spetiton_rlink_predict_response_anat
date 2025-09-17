import numpy as np

# Models
from sklearn.base import clone
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
import sklearn.svm as svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.pipeline import make_pipeline


from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

from ml_utils import GroupFeatureTransformer

def make_models(n_jobs_grid_search, cv_val,
                residualization_formula=None,
                residualizer_estimator=None,
                roi_groups=None):
    """_summary_

    Parameters
    ----------
    n_jobs_grid_search : int
        Nb jobs for grd search
    residualization_formula : str, optional
        if string is provided use residualization, and labels model with this
        string, by default None
    residualizer_estimator :  ResidualizerEstimator()
        The residualizer

    Returns
    -------
    dict
        Dictionary of models
    """
    
    # Param grd for MLP
    mlp_param_grid = {"hidden_layer_sizes":
                      [(100, ), (50, ), (25, ), (10, ), (5, ),         # 1 hidden layer
                       (100, 50, ), (50, 25, ), (25, 10,
                                                 # 2 hidden layers
                                                 ), (10, 5, ),
                          # 3 hidden layers
                       (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )],
                      "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}


    # Models models_backbones to be completed with residualizer, 
    models_backbones = {
        'model-lrl2cv':[
            preprocessing.StandardScaler(),
            # preprocessing.MinMaxScaler(),
            GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                         {'C': 10. ** np.arange(-3, 1)},
                         cv=cv_val, n_jobs=n_jobs_grid_search)],

        'model-lrenetcv':[
            preprocessing.StandardScaler(),
            # preprocessing.MinMaxScaler(),
            GridSearchCV(estimator=lm.SGDClassifier(loss='log_loss',
                                                    penalty='elasticnet',
                                                    fit_intercept=False, class_weight='balanced'),
                         param_grid={'alpha': 10. ** np.arange(-1, 3),
                                     'l1_ratio': [.1, .5, .9]},
                         cv=cv_val, n_jobs=n_jobs_grid_search)],

        'model-svmrbfcv':[
            # preprocessing.StandardScaler(),
            preprocessing.MinMaxScaler(),
            GridSearchCV(svm.SVC(class_weight='balanced', probability=True),
                         # {'kernel': ['poly', 'rbf'], 'C': 10. ** np.arange(-3, 3)},
                         {'kernel': ['rbf'], 'C': 10. ** np.arange(-1, 2)},
                         cv=cv_val, n_jobs=n_jobs_grid_search)],

        'model-forestcv':[
            # preprocessing.StandardScaler(),
            preprocessing.MinMaxScaler(),
            GridSearchCV(RandomForestClassifier(random_state=1, class_weight='balanced'),
                         {"n_estimators": [10, 100]},
                         cv=cv_val, n_jobs=n_jobs_grid_search)],

        'model-gbcv':[
            preprocessing.MinMaxScaler(),
            GridSearchCV(estimator=GradientBoostingClassifier(random_state=1),
                         param_grid={"n_estimators": [10, 100]},
                         cv=cv_val, n_jobs=n_jobs_grid_search)],

    #     'mlp_cv':[
    #         # preprocessing.StandardScaler(),
    #         preprocessing.MinMaxScaler(),
    #         GridSearchCV(estimator=MLPClassifier(random_state=1, max_iter=200, tol=0.01),
    #                      param_grid=mlp_param_grid,
    #                      cv=cv_val, n_jobs=n_jobs_grid_search)]
    
            'model-grpRoiLda+lrl2':[
                GroupFeatureTransformer(roi_groups,  "lda"),
                preprocessing.StandardScaler(),
                GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'),
                            {'C': 10. ** np.arange(-3, 1)},
                            cv=cv_val, n_jobs=5, scoring='balanced_accuracy')],
    }

    if residualization_formula:
        models = {model_name_prefix + "_resid-%s" % residualization_formula:\
            make_pipeline(* [residualizer_estimator] + model_steps)
                        for model_name_prefix, model_steps
                        in models_backbones.items()}

    else:
        models = {model_name_prefix:\
            make_pipeline(*model_steps)
                    for model_name_prefix, model_steps
                    in models_backbones.items()}

    return models
