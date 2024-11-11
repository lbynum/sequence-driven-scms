import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm
from econml.grf import CausalForest
from econml.dml import LinearDML, CausalForestDML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from econml.dr import ForestDRLearner, LinearDRLearner
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from catenets.models.jax import SNet, FlexTENet, OffsetNet, TNet, SNet1, SNet2, SNet3, DRNet, RANet, PWNet, RNet, XNet

from utils import add_normalized_colums


def format_prediction_results(
    estimated_cates,
    cate_lower_bounds,
    cate_upper_bounds,
    estimated_ate,
    ate_lower_bound,
    ate_upper_bound,
    estimated_ites,
    ite_lower_bounds,
    ite_upper_bounds
):
    estimate_dict = {
        'ate': estimated_ate,
        'ate_lower_bound': ate_lower_bound,
        'ate_upper_bound': ate_upper_bound,
        'cate': estimated_cates,
        'cate_lower_bounds': cate_lower_bounds,
        'cate_upper_bounds': cate_upper_bounds,
        'ite': estimated_ites,
        'ite_lower_bounds': ite_lower_bounds,
        'ite_upper_bounds': ite_upper_bounds,
    }
    return estimate_dict


def predict_forest_dr(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.dr.LinearDRLearner.html
    model = ForestDRLearner(discrete_outcome=False, n_jobs=None)
    model.fit(Y=y, T=z, X=x, W=None)

    # get ATE estimate
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = model.ate_interval(X=x)

    # get CATE estimates
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = model.effect_interval(X=x)

    # there are no explicit ITE estimates
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_linear_dr(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.dr.LinearDRLearner.html
    model = LinearDRLearner(discrete_outcome=False)
    model.fit(Y=y, T=z, X=x, W=None)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = model.ate_interval(X=x)
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = model.effect_interval(X=x)
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_linear_dml(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.dml.LinearDML.html
    model = LinearDML(discrete_treatment=True, discrete_outcome=False)
    model.fit(Y=y, T=z, X=x, W=None)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = model.ate_interval(X=x)
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = model.effect_interval(X=x)
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_causalforest_dml(x, z, y):
    # https://econml.azurewebsites.net/_autosummary/econml.dml.CausalForestDML.html
    model = CausalForestDML(discrete_treatment=True, discrete_outcome=False, n_jobs=None)
    model.fit(Y=y, T=z, X=x, W=None)
    estimated_ate = model.ate(X=x)
    ate_lower_bound, ate_upper_bound = model.ate_interval(X=x)
    estimated_cates = model.effect(X=x)
    cate_lower_bounds, cate_upper_bounds = model.effect_interval(X=x)
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_causal_forest(x, z, y):
    model = CausalForest(n_jobs=None)
    model.fit(x, z, y)
    estimated_cates, cate_lower_bounds, cate_upper_bounds = model.predict(X=x, interval=True)
    estimated_ate = estimated_cates.mean()
    ate_lower_bound = None
    ate_upper_bound = None
    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_naive_linear_regression(x, z, y):
    x_0 = np.expand_dims(np.zeros_like(z), -1)
    x_1 = np.expand_dims(np.ones_like(z), -1)
    model = LinearRegression().fit(X=np.expand_dims(z, -1), y=y)

    estimated_cates = model.predict(X=x_1) - model.predict(X=x_0)
    cate_lower_bounds = None
    cate_upper_bounds = None

    estimated_ate = model.coef_[0]
    ate_lower_bound = None
    ate_upper_bound = None

    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_linear_regression(x, z, y):
    x_full = np.c_[x, z]
    x_0 = np.c_[x, np.zeros_like(z)]
    x_1 = np.c_[x, np.ones_like(z)]
    model = LinearRegression().fit(X=x_full, y=y)

    estimated_cates = model.predict(X=x_1) - model.predict(X=x_0)
    cate_lower_bounds = None
    cate_upper_bounds = None

    estimated_ate = model.coef_[-1]
    ate_lower_bound = None
    ate_upper_bound = None

    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_naive_random_forest(x, z, y):
    x_0 = np.expand_dims(np.zeros_like(z), -1)
    x_1 = np.expand_dims(np.ones_like(z), -1)
    model = RandomForestRegressor().fit(X=np.expand_dims(z, -1), y=y)

    estimated_cates = model.predict(X=x_1) - model.predict(X=x_0)
    cate_lower_bounds = None
    cate_upper_bounds = None

    estimated_ate = np.mean(estimated_cates)
    ate_lower_bound = None
    ate_upper_bound = None

    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_random_forest(x, z, y):
    x_full = np.c_[x, z]
    x_0 = np.c_[x, np.zeros_like(z)]
    x_1 = np.c_[x, np.ones_like(z)]
    model = RandomForestRegressor().fit(X=x_full, y=y)

    estimated_cates = model.predict(X=x_1) - model.predict(X=x_0)
    cate_lower_bounds = None
    cate_upper_bounds = None

    estimated_ate = np.mean(estimated_cates)
    ate_lower_bound = None
    ate_upper_bound = None

    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


def predict_catenet(model_name, x, z, y):
    catenet_lookup = dict(
        SNet=SNet,
        FlexTENet=FlexTENet,
        OffsetNet=OffsetNet,
        TNet=TNet,
        SNet1=SNet1,
        SNet2=SNet2,
        SNet3=SNet3,
        DRNet=DRNet,
        RANet=RANet,
        PWNet=PWNet,
        RNet=RNet,
        XNet=XNet
    )
    model = catenet_lookup[model_name]()
    model.fit(X=x, y=y, w=z)

    estimated_cates = model.predict(X=x)
    cate_lower_bounds = None
    cate_upper_bounds = None

    estimated_ate = np.mean(estimated_cates)
    ate_lower_bound = None
    ate_upper_bound = None

    estimated_ites = None
    ite_lower_bounds = None
    ite_upper_bounds = None

    return format_prediction_results(
        estimated_ate=estimated_ate,
        ate_lower_bound=ate_lower_bound,
        ate_upper_bound=ate_upper_bound,
        estimated_cates=estimated_cates,
        cate_lower_bounds=cate_lower_bounds,
        cate_upper_bounds=cate_upper_bounds,
        estimated_ites=estimated_ites,
        ite_lower_bounds=ite_lower_bounds,
        ite_upper_bounds=ite_upper_bounds
    )


METHOD_LOOKUP = dict(
    SNet=lambda x, z, y: predict_catenet('SNet', x, z, y),
    FlexTENet=lambda x, z, y: predict_catenet('FlexTENet', x, z, y),
    OffsetNet=lambda x, z, y: predict_catenet('OffsetNet', x, z, y),
    TNet=lambda x, z, y: predict_catenet('TNet', x, z, y),
    SNet1=lambda x, z, y: predict_catenet('SNet1', x, z, y),
    SNet2=lambda x, z, y: predict_catenet('SNet2', x, z, y),
    SNet3=lambda x, z, y: predict_catenet('SNet3', x, z, y),
    DRNet=lambda x, z, y: predict_catenet('DRNet', x, z, y),
    RANet=lambda x, z, y: predict_catenet('RANet', x, z, y),
    PWNet=lambda x, z, y: predict_catenet('PWNet', x, z, y),
    RNet=lambda x, z, y: predict_catenet('RNet', x, z, y),
    XNet=lambda x, z, y: predict_catenet('XNet', x, z, y),
    CausalForest=predict_causal_forest,
    NaiveLinReg=predict_naive_linear_regression,
    NaiveRF=predict_naive_random_forest,
    LinReg=predict_linear_regression,
    RF=predict_random_forest,
    CausalForestDML=predict_causalforest_dml,
    LinearDML=predict_linear_dml,
    ForestDR=predict_forest_dr,
    LinearDR=predict_linear_dr
)


def run_method(args):
    df_path, method_name, covariate_names, treatment_name, y_name, outcome_index, possible_outcome_values, treatment_index, possible_treatment_values = args
    data_df = pd.read_csv(df_path)
    data_df = add_normalized_colums(data_df, outcome_index, possible_outcome_values, treatment_index, possible_treatment_values)

    x = data_df[covariate_names].values
    z = data_df[treatment_name].values
    y = data_df[y_name].values

    prediction_result = METHOD_LOOKUP[method_name](x=x, z=z, y=y)

    y1 = data_df[f'{y_name}|do({treatment_name}=1)'].values
    y0 = data_df[f'{y_name}|do({treatment_name}=0)'].values

    prediction_result.update({
        'df_path': df_path,
        'method_name': method_name,
        'covariate_names': tuple(covariate_names),
        'treatment_name': treatment_name,
        'y_name': y_name,
        'dataset_size': len(y),
        'treatment': z,
        'y0': y0,
        'y1': y1,
        'y': y
    })

    return prediction_result


def run_estimation_methods(
    dataset_paths,
    method_names,
    covariate_names,
    treatment_name,
    y_name,
    outcome_index,
    possible_outcome_values,
    treatment_index,
    possible_treatment_values,
    num_processes=6
):
    tasks = (
        (
            df_path,
            method_name,
            covariate_names,
            treatment_name,
            y_name,
            outcome_index,
            possible_outcome_values,
            treatment_index,
            possible_treatment_values
        )
        for df_path in dataset_paths
        for method_name in method_names
    )

    with mp.Pool(processes=num_processes) as pool:
        result = [
            item
            for item in tqdm(pool.imap_unordered(run_method, tasks), total=len(dataset_paths)*len(method_names))
        ]
        result_df = pd.DataFrame(result)

    return result_df
