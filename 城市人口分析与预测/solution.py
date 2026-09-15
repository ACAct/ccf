# -*- coding: utf-8 -*-
"""
城市人口分析与预测 —— 完整技术实现（按优化流水线）
=====================================================
数据清洗 → 保证城市-年份连续 → 构造高质量人口特征 → 特征精简(170→~70)
→ 构造历史预测样本 → 严格 Walk-forward OOF → LightGBM 参数优化
→ Density Ratio 加入 → Trend/Ridge/CatBoost 辅助 → 获得所有模型 OOF
→ 真实人口 MSE 评价 → NNLS 优化 Ensemble 权重 → 多步递推回测
→ 确定最终模型 → 全部历史数据训练 → 2022→2023 递推 → 后处理 → submission.csv

目标：预测 40 个城市（city1~city40）2023 年总人口（常住人口，万人）。
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from scipy.optimize import nnls

import lightgbm as lgb
from catboost import CatBoostRegressor

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 200)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 一、数据读取
# ----------------------------------------------------------------------
def read_population():
    df = pd.read_excel(os.path.join(DATA_DIR, '人口规模.xlsx'))
    df = df.rename(columns={'城市名称': 'city_id', '年份': 'year',
                            '常住人口（万人）': 'population',
                            '户籍人口（万人）': 'hukou_population'})
    return df[['city_id', 'year', 'population', 'hukou_population']]


def read_density():
    df = pd.read_excel(os.path.join(DATA_DIR, '人口密度.xlsx'))
    df = df.rename(columns={'城市名称': 'city_id', '年份': 'year',
                            '人口密度（人/平方公里）': 'density'})
    return df[['city_id', 'year', 'density']]


def read_urbanization():
    df = pd.read_excel(os.path.join(DATA_DIR, '城镇化率.xlsx'))
    df = df.rename(columns={'城市名称': 'city_id', '年份': 'year',
                            'urbanizationRate': 'urbanization'})
    return df[['city_id', 'year', 'urbanization']]


def read_employment():
    xl = pd.ExcelFile(os.path.join(DATA_DIR, '就业信息.xlsx'))
    unemp = xl.parse('城镇失业率').rename(
        columns={'城市名称': 'city_id', '年份': 'year', 'unemploymentRate': 'unemployment'})
    emp = xl.parse('从业人员数').rename(
        columns={'城市名称': 'city_id', '年份': 'year', 'employeesNumber': 'employees'})
    ind = xl.parse('第一、二、三产业就业人数').rename(
        columns={'城市名称': 'city_id', '年份': 'year',
                 'pi_Employment': 'pi_employment',
                 'si_Employment': 'si_employment',
                 'ti_Employment': 'ti_employment'})
    return (unemp[['city_id', 'year', 'unemployment']],
            emp[['city_id', 'year', 'employees']],
            ind[['city_id', 'year', 'pi_employment', 'si_employment', 'ti_employment']])


def read_wage():
    raw = pd.read_excel(os.path.join(DATA_DIR, '工资水平.xlsx'), sheet_name='职工平均工资')
    raw = raw.rename(columns={raw.columns[0]: 'year'})
    raw['year'] = raw['year'].astype(str).str.replace('年', '').astype(int)
    return raw.melt(id_vars='year', var_name='city_id', value_name='wage')[['city_id', 'year', 'wage']]


def read_age():
    df = pd.read_excel(os.path.join(DATA_DIR, '年龄结构.xlsx'))
    df = df.rename(columns={'城市名称': 'city_id', '年份': 'year',
                            '0-14': 'age_0_14', '15-64': 'age_15_64', '65+': 'age_65_plus'})
    return df[['city_id', 'year', 'age_0_14', 'age_15_64', 'age_65_plus']]


def _read_wide_living(sheet, value_col):
    raw = pd.read_excel(os.path.join(DATA_DIR, '生活水平.xlsx'), sheet_name=sheet, header=None)
    years = pd.to_numeric(raw.iloc[1, 1:], errors='coerce').astype(int).values
    cities = raw.iloc[2:, 0].astype(str).str.strip().values
    vals = raw.iloc[2:, 1:].values
    n_city, n_year = vals.shape
    rows = [(cities[i], years[j], vals[i, j]) for i in range(n_city) for j in range(n_year)]
    df = pd.DataFrame(rows, columns=['city_id', 'year', value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    return df


def read_living():
    sheets = {'人均可支配收入': 'income', '人均消费支出': 'consumption',
              '城镇居民消费支出': 'towner_consumption', '农村居民消费支出': 'rural_consumption',
              '城镇居民人均收入': 'towner_income', '农村居民人均收入': 'rural_income'}
    return {col: _read_wide_living(sh, col) for sh, col in sheets.items()}


def load_all():
    frames = [read_population(), read_density(), read_urbanization()]
    unemp, emp, ind = read_employment()
    frames += [unemp, emp, ind, read_wage(), read_age()]
    frames += list(read_living().values())
    train = frames[0]
    for f in frames[1:]:
        train = train.merge(f, on=['city_id', 'year'], how='outer')
    return train


# ----------------------------------------------------------------------
# 二、数据清洗 + 保证城市-年份连续
# ----------------------------------------------------------------------
def clean(train):
    train = train.copy()
    train['city_id'] = train['city_id'].astype(str).str.strip()
    train['city_id'] = train['city_id'].str.replace('ctiy', 'city', regex=False)  # 修正录入错误
    train['year'] = pd.to_numeric(train['year'], errors='coerce').astype('Int64')
    dup = train.duplicated(subset=['city_id', 'year']).sum()
    if dup:
        print(f'[清洗] 发现重复记录 {dup} 条，按 (city,year) 聚合取均值')
        train = train.groupby(['city_id', 'year'], as_index=False).mean(numeric_only=True)
    return train.sort_values(['city_id', 'year']).reset_index(drop=True)


def ensure_continuity(train):
    """保证每个城市的年份连续（缺失年份补为 NaN），使滞后特征按正确年份计算"""
    parts = []
    for city, sub in train.groupby('city_id', sort=False):
        sub = sub.sort_values('year')
        full = pd.DataFrame({'year': range(int(sub['year'].min()), int(sub['year'].max()) + 1)})
        full['city_id'] = city
        full = full.merge(sub, on=['city_id', 'year'], how='left')
        parts.append(full)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------
# 三、构造高质量人口特征（170 个）
# ----------------------------------------------------------------------
RAW_COLS = ['population', 'hukou_population', 'density', 'urbanization', 'unemployment',
            'employees', 'pi_employment', 'si_employment', 'ti_employment', 'wage',
            'income', 'consumption', 'towner_consumption', 'rural_consumption',
            'towner_income', 'rural_income', 'age_0_14', 'age_15_64', 'age_65_plus']


def build_features(df):
    panel = df.copy().sort_values(['city_id', 'year']).reset_index(drop=True)

    # 低频变量（年龄结构）向前填充
    for c in ['age_0_14', 'age_15_64', 'age_65_plus']:
        panel[c] = panel.groupby('city_id')[c].ffill()

    # 就业总人数、产业占比
    panel['emp_total'] = panel['pi_employment'] + panel['si_employment'] + panel['ti_employment']
    panel['pi_share'] = panel['pi_employment'] / panel['emp_total'].replace(0, np.nan)
    panel['si_share'] = panel['si_employment'] / panel['emp_total'].replace(0, np.nan)
    panel['ti_share'] = panel['ti_employment'] / panel['emp_total'].replace(0, np.nan)

    # 年龄结构占比与抚养比
    panel['age_total'] = panel['age_0_14'] + panel['age_15_64'] + panel['age_65_plus']
    panel['child_share'] = panel['age_0_14'] / panel['age_total'].replace(0, np.nan)
    panel['working_share'] = panel['age_15_64'] / panel['age_total'].replace(0, np.nan)
    panel['old_share'] = panel['age_65_plus'] / panel['age_total'].replace(0, np.nan)
    panel['child_dep'] = panel['age_0_14'] / panel['age_15_64'].replace(0, np.nan)
    panel['old_dep'] = panel['age_65_plus'] / panel['age_15_64'].replace(0, np.nan)
    panel['total_dep'] = (panel['age_0_14'] + panel['age_65_plus']) / panel['age_15_64'].replace(0, np.nan)

    # 常住-户籍：人口净流入代理指标
    panel['net_inflow'] = panel['population'] - panel['hukou_population']
    panel['pop_hukou_ratio'] = panel['population'] / panel['hukou_population'].replace(0, np.nan)

    # 对全部变量构造滞后与增长率
    lag_vars = list(dict.fromkeys(RAW_COLS + [
        'emp_total', 'pi_share', 'si_share', 'ti_share',
        'age_total', 'child_share', 'working_share', 'old_share',
        'child_dep', 'old_dep', 'total_dep', 'net_inflow', 'pop_hukou_ratio']))
    panel = panel.sort_values(['city_id', 'year']).reset_index(drop=True)
    g = panel.groupby('city_id', sort=False)
    for c in lag_vars:
        for k in (1, 2, 3):
            panel[f'{c}_lag{k}'] = g[c].shift(k)
        panel[f'{c}_growth'] = panel[c] / panel[f'{c}_lag1'] - 1.0

    # 人口专属特征
    p = panel.groupby('city_id', sort=False)['population']
    for k in (4, 5, 6):
        panel[f'population_lag{k}'] = p.shift(k)
    panel['pop_growth'] = panel['population'] / panel['population_lag1'] - 1.0  # Growth_t
    panel['pop_growth_last'] = panel['population_lag1'] / panel['population_lag2'] - 1.0
    panel['pop_growth_lag2'] = panel['population_lag2'] / panel['population_lag3'] - 1.0
    panel['pop_growth_lag3'] = panel['population_lag3'] / panel['population_lag4'] - 1.0
    panel['pop_growth_lag4'] = panel['population_lag4'] / panel['population_lag5'] - 1.0
    panel['pop_growth_lag5'] = panel['population_lag5'] / panel['population_lag6'] - 1.0
    panel['pop_growth_avg3'] = panel[['pop_growth', 'pop_growth_last', 'pop_growth_lag2']].mean(axis=1)
    panel['pop_growth_avg5'] = panel[['pop_growth', 'pop_growth_last', 'pop_growth_lag2',
                                      'pop_growth_lag3', 'pop_growth_lag4']].mean(axis=1)

    return panel.replace([np.inf, -np.inf], np.nan)


def add_target(panel):
    """预测目标：下一年人口增长率 Growth_{t+1}"""
    panel = panel.sort_values(['city_id', 'year']).reset_index(drop=True)
    panel['target'] = panel.groupby('city_id', sort=False)['pop_growth'].shift(-1)
    return panel


# ----------------------------------------------------------------------
# 四、模型定义
# ----------------------------------------------------------------------
def trend_predict(X):
    """历史趋势模型：近 5 年逐年增长率的中位数"""
    parts = np.column_stack([
        X['pop_growth'].values, X['pop_growth_last'].values,
        X['pop_growth_lag2'].values, X['pop_growth_lag3'].values,
        X['pop_growth_lag4'].values]).astype(float)
    return np.nanmedian(parts, axis=1)


def density_ratio_predict(X):
    """密度比例模型：用当年人口密度增长率预测下一年人口增长率（密度作为领先指标）"""
    return X['density_growth'].values


def make_lgbm(**kw):
    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=5,
                  min_child_samples=8, subsample=0.85, subsample_freq=1,
                  colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=1.0,
                  random_state=RANDOM_STATE, verbose=-1)
    params.update(kw)
    return lgb.LGBMRegressor(**params)


def make_cat():
    return CatBoostRegressor(iterations=500, learning_rate=0.03, depth=5, l2_leaf_reg=3.0,
                             random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False)


def make_ridge(alpha=10.0):
    return Ridge(alpha=alpha)


def fit_predict_model(model, X_tr, y_tr, X_te):
    if isinstance(model, Ridge):
        imp = SimpleImputer(strategy='median')
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(imp.fit_transform(X_tr))
        X_te_s = sc.transform(imp.transform(X_te))
        model.fit(X_tr_s, y_tr)
        return model.predict(X_te_s)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def predict_growth_row(X_tr, y_tr, X_te, specs):
    """用给定模型集在 X_te 上预测增长率，返回 {name: pred}"""
    out = {}
    for name, mk in specs:
        if name == 'trend':
            out[name] = trend_predict(X_te)
        elif name == 'density':
            out[name] = density_ratio_predict(X_te)
        elif name == 'ridge':
            out[name] = fit_predict_model(make_ridge(), X_tr, y_tr, X_te)
        else:
            out[name] = fit_predict_model(mk(), X_tr, y_tr, X_te)
    return out


# ----------------------------------------------------------------------
# 五、特征精简（LightGBM 特征重要性，170 -> ~70）
# ----------------------------------------------------------------------
def select_features(samples, feature_cols, n_keep=70):
    model = make_lgbm()
    model.fit(samples[feature_cols], samples['target'])
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    # 剔除缺失率过高的特征（>60%）
    missing_rate = samples[feature_cols].isna().mean()
    keep = [c for c in imp.index if missing_rate[c] <= 0.6]
    return keep[:n_keep], imp


# ----------------------------------------------------------------------
# 六、LightGBM 参数优化（小网格，用 Walk-forward OOF 的 MSE 评价）
# ----------------------------------------------------------------------
def tune_lgbm(samples, feature_cols, val_years):
    grid = [
        dict(num_leaves=15, min_child_samples=8, learning_rate=0.03, colsample_bytree=0.8),
        dict(num_leaves=31, min_child_samples=8, learning_rate=0.03, colsample_bytree=0.8),
        dict(num_leaves=15, min_child_samples=15, learning_rate=0.03, colsample_bytree=0.8),
        dict(num_leaves=15, min_child_samples=8, learning_rate=0.05, colsample_bytree=0.8),
        dict(num_leaves=15, min_child_samples=8, learning_rate=0.03, colsample_bytree=0.9),
        dict(num_leaves=31, min_child_samples=15, learning_rate=0.03, colsample_bytree=0.9),
        dict(num_leaves=15, min_child_samples=8, learning_rate=0.05, colsample_bytree=0.9),
        dict(num_leaves=31, min_child_samples=8, learning_rate=0.05, colsample_bytree=0.9),
    ]
    best_params, best_score = None, np.inf
    for kw in grid:
        oof = np.full(len(samples), np.nan)
        for vy in val_years:
            tr_idx = samples['year'] < vy
            va_idx = samples['year'] == vy
            if va_idx.sum() == 0 or tr_idx.sum() == 0:
                continue
            m = make_lgbm(**kw)
            m.fit(samples.loc[tr_idx, feature_cols], samples.loc[tr_idx, 'target'])
            oof[va_idx.to_numpy()] = m.predict(samples.loc[va_idx, feature_cols])
        p_now = samples['population'].values
        p_next = p_now * (1.0 + samples['target'].values)
        pred_pop = p_now * (1.0 + oof)
        mask = np.isfinite(pred_pop) & np.isfinite(p_next)
        mse = mean_squared_error(p_next[mask], pred_pop[mask])
        if mse < best_score:
            best_score, best_params = mse, kw
    print(f'[调参] 最佳 LightGBM 参数: {best_params} (OOF MSE={best_score:.2f})')
    return best_params


# ----------------------------------------------------------------------
# 七、主流程
# ----------------------------------------------------------------------
def main():
    print('=' * 70)
    print('一、数据清洗')
    train = clean(load_all())
    train = ensure_continuity(train)
    print(f'  清洗+连续化后: {train.shape}, 城市数: {train["city_id"].nunique()}')

    print('二、构造高质量人口特征')
    panel = add_target(build_features(train))
    feature_cols = [c for c in panel.columns
                    if c not in ('city_id', 'year', 'target', 'population_growth')]
    feature_cols = [c for c in feature_cols if panel[c].notna().any()]
    print(f'  原始特征数: {len(feature_cols)}')

    samples = panel.dropna(subset=['target']).reset_index(drop=True)
    lo, hi = samples['target'].quantile([0.005, 0.995])
    samples['target'] = samples['target'].clip(lo, hi)
    print(f'  训练样本: {len(samples)}')

    print('三、特征精简（170 -> ~70，LightGBM 特征重要性）')
    feature_cols, imp = select_features(samples, feature_cols, n_keep=70)
    print(f'  精简后特征数: {len(feature_cols)}')
    print('  Top 10 特征:', list(imp.head(10).index))

    print('四、LightGBM 参数优化')
    val_years = sorted(samples['year'].unique())[-6:]
    lgb_params = tune_lgbm(samples, feature_cols, val_years)

    print('五、模型训练 + Walk-forward OOF')
    specs = [('lgb', lambda: make_lgbm(**lgb_params)), ('cat', make_cat),
             ('ridge', make_ridge), ('trend', None), ('density', None)]
    names = [n for n, _ in specs]
    oof = {n: np.full(len(samples), np.nan) for n in names}
    for vy in val_years:
        tr_idx = samples['year'] < vy
        va_idx = samples['year'] == vy
        if va_idx.sum() == 0 or tr_idx.sum() == 0:
            continue
        preds = predict_growth_row(samples.loc[tr_idx, feature_cols],
                                   samples.loc[tr_idx, 'target'],
                                   samples.loc[va_idx, feature_cols], specs)
        for n in names:
            oof[n][va_idx.to_numpy()] = preds[n]

    print('六、真实人口 MSE 评价')
    p_now = samples['population'].values
    p_next = p_now * (1.0 + samples['target'].values)
    for n in names:
        pred_pop = p_now * (1.0 + oof[n])
        mask = np.isfinite(pred_pop) & np.isfinite(p_next)
        print(f'    {n:8s}: OOF 人口 MSE = {mean_squared_error(p_next[mask], pred_pop[mask]):10.2f}')

    print('七、NNLS 优化 Ensemble 权重')
    # 密度比例模型 NaN 过多且效果差（仅报告其 MSE），不参与融合
    ensemble_names = [n for n in names if n != 'density']
    stack_valid = np.isfinite(p_next) & np.all([np.isfinite(oof[n]) for n in ensemble_names], axis=0)
    X_pop = p_now[stack_valid, None] * (1.0 + np.column_stack([oof[n][stack_valid] for n in ensemble_names]))
    y_pop = p_next[stack_valid]
    w_nn, _ = nnls(X_pop, y_pop)
    weights = w_nn / w_nn.sum()
    for n, w in zip(ensemble_names, weights):
        print(f'    {n:8s}: w = {w:.3f}')
    print(f'  融合后 OOF 人口 MSE = {mean_squared_error(y_pop, X_pop @ weights):.2f}')

    # 八、多步递推回测
    print('八、多步递推回测（留出 2020、2021）')

    def recursive_forecast(base_df, start_year, n_steps, cutoff_year):
        """从 start_year 起，用 cutoff_year 及以前数据训练，递推 n_steps 步"""
        tr = samples[samples['year'] < cutoff_year]
        X_tr = tr[feature_cols]; y_tr = tr['target']
        base = base_df.copy()
        cur_panel = build_features(base)
        preds = {}
        for step in range(n_steps):
            yr = start_year + step + 1
            cur_year = start_year + step
            row = cur_panel[cur_panel['year'] == cur_year].sort_values('city_id').reset_index(drop=True)
            g = predict_growth_row(X_tr, y_tr, row[feature_cols], specs)
            # 用融合权重（NaN 安全）
            growth = np.nansum(np.column_stack([g[n] for n in ensemble_names]) * weights, axis=1)
            preds[yr] = row['population'].values * (1.0 + growth)
            # 把预测人口填入，重新生成特征供下一步使用
            for city, p in zip(row['city_id'], preds[yr]):
                mask = (base['city_id'] == city) & (base['year'] == yr)
                if mask.any():
                    base.loc[mask, 'population'] = p
                else:
                    base = pd.concat([base, pd.DataFrame(
                        [{'city_id': city, 'year': yr, 'population': p}])], ignore_index=True)
            cur_panel = build_features(base)
        return preds

    # 回测：只用 2019 及以前数据（模拟过去，避免使用未来特征）
    preds_bt = recursive_forecast(train[train['year'] <= 2019], 2019, 2, cutoff_year=2019)
    true_2020 = panel[panel['year'] == 2020].sort_values('city_id')['population'].values
    true_2021 = panel[panel['year'] == 2021].sort_values('city_id')['population'].values
    print(f'  回测 1 步(2020) 人口 MSE = {mean_squared_error(true_2020, preds_bt[2020]):.2f}')
    print(f'  回测 2 步(2021) 人口 MSE = {mean_squared_error(true_2021, preds_bt[2021]):.2f}')

    # 九、全部历史数据训练 + 2022→2023 递推（非人口特征可用到 2022）
    print('九、全部历史数据训练，2022→2023 递推预测')
    preds = recursive_forecast(train, 2021, 2, cutoff_year=2021)
    p_2022, p_2023 = preds[2022], preds[2023]

    # 十、后处理
    print('十、后处理')
    p_2023 = np.clip(p_2023, 0, None)
    print(f'  预测值范围: {p_2023.min():.2f} ~ {p_2023.max():.2f}（非负检查通过）')

    # 十一、生成提交文件
    print('十一、生成提交文件 submission.csv')
    cities = sorted(train['city_id'].unique())  # 与 p_2023 顺序一致（字典序）
    sub = pd.DataFrame({'city_id': cities, 'year': 2023, 'pred': np.round(p_2023, 2)})
    print(sub.to_string(index=False))
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submission.csv')
    try:
        sub.to_csv(out_path, index=False)
        print(f'  已写入: {out_path}')
    except PermissionError:
        print(f'  [警告] 无法写入 {out_path}（文件被占用，请关闭 Excel/其它程序后重试）')

    print('完成。')


if __name__ == '__main__':
    main()
