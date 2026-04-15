import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Simplified CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('[fonts.googleapis.com](https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap)');

:root {
    --bg: #0f0f0f;
    --card: #1a1a1a;
    --border: #2a2a2a;
    --accent: #3b82f6;
    --success: #22c55e;
    --warning: #eab308;
    --danger: #ef4444;
    --text: #fafafa;
    --muted: #737373;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem; max-width: 1400px; }

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Header */
.header {
    padding: 1.5rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}
.header p {
    color: var(--muted);
    margin: 0.25rem 0 0 0;
    font-size: 0.9rem;
}

/* Steps */
.steps {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
}
.step {
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 500;
}
.step.done { background: rgba(34,197,94,0.15); color: #22c55e; }
.step.active { background: var(--accent); color: white; }
.step.pending { background: var(--card); color: var(--muted); border: 1px solid var(--border); }

/* Metrics */
.metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
}
.metric {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.25rem;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
}
.badge-blue { background: rgba(59,130,246,0.15); color: #3b82f6; }
.badge-green { background: rgba(34,197,94,0.15); color: #22c55e; }
.badge-yellow { background: rgba(234,179,8,0.15); color: #eab308; }
.badge-red { background: rgba(239,68,68,0.15); color: #ef4444; }

/* Streamlit Overrides */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput input,
.stTextInput input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

div[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--border) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────
defaults = {
    'step': 0, 'problem_type': None, 'df': None, 'target': None,
    'features': None, 'df_clean': None, 'selected_features': None,
    'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
    'model': None, 'model_name': None, 'k_folds': 5, 'results': {},
    'outlier_indices': [], 'scaler': None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Helpers ────────────────────────────────────────────────────────────────
STEPS = ["Problem", "Data", "EDA", "Engineering", "Features", "Split", "Model", "Training", "Metrics", "Tuning"]

def render_steps(current):
    html = '<div class="steps">'
    for i, name in enumerate(STEPS):
        cls = "done" if i < current else "active" if i == current else "pending"
        html += f'<div class="step {cls}">{i+1}. {name}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def metric_cards(data):
    html = '<div class="metrics">'
    for label, value in data.items():
        html += f'<div class="metric"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def next_step():
    st.session_state.step += 1

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(26,26,26,0.5)',
    font=dict(family='Inter', color='#a3a3a3', size=12),
    xaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a'),
    yaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a'),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=['#3b82f6', '#22c55e', '#eab308', '#ef4444', '#8b5cf6', '#ec4899'],
)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>Pipelytics AI</h1>
    <p>End-to-end machine learning workflow</p>
</div>
""", unsafe_allow_html=True)

render_steps(st.session_state.step)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 0: Problem Type
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    st.subheader("Select Problem Type")
    
    col1, col2 = st.columns(2)
    with col1:
        problem = st.radio("What type of ML problem?", ["Regression", "Classification"], horizontal=True)
    with col2:
        desc = "Predict continuous values" if problem == "Regression" else "Classify into categories"
        st.info(f"**{problem}**: {desc}")
    
    if st.button("Continue →"):
        st.session_state.problem_type = problem
        next_step()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Data Input
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    st.subheader("Data Input")
    
    data_src = st.radio("Data source:", ["Boston Housing (built-in)", "Upload CSV"], horizontal=True)
    
    if data_src == "Boston Housing (built-in)":
        np.random.seed(42)
        n = 506
        df = pd.DataFrame({
            'CRIM': np.abs(np.random.exponential(3.6, n)),
            'ZN': np.abs(np.random.exponential(11, n)),
            'INDUS': np.abs(np.random.normal(11, 7, n)),
            'CHAS': np.random.choice([0,1], n, p=[0.93,0.07]),
            'NOX': np.abs(np.random.normal(0.55, 0.12, n)),
            'RM': np.abs(np.random.normal(6.28, 0.7, n)),
            'AGE': np.clip(np.random.normal(68, 28, n), 0, 100),
            'DIS': np.abs(np.random.exponential(3.8, n)),
            'RAD': np.random.choice(range(1,25), n),
            'TAX': np.abs(np.random.normal(408, 168, n)),
            'PTRATIO': np.abs(np.random.normal(18.4, 2.1, n)),
            'B': np.clip(np.random.normal(354, 89, n), 0, 396),
            'LSTAT': np.abs(np.random.normal(12.6, 7.1, n)),
            'MEDV': np.abs(np.random.normal(22.5, 9.2, n))
        })
        st.session_state.df = df
        st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"Uploaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("---")
        st.subheader("Target & Features")
        
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Target", df.columns.tolist(), index=len(df.columns)-1)
        with col2:
            feat_cols = [c for c in df.columns if c != target]
            features = st.multiselect("Features", feat_cols, default=feat_cols)
        
        if features:
            metric_cards({
                "Rows": f"{df.shape[0]:,}",
                "Features": str(len(features)),
                "Target": target,
            })
        
        if st.button("Continue →") and features:
            st.session_state.target = target
            st.session_state.features = features
            st.session_state.df_clean = df[features + [target]].copy()
            next_step()
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: EDA
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    df = st.session_state.df_clean
    target = st.session_state.target
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    st.subheader("Exploratory Data Analysis")
    
    tabs = st.tabs(["Summary", "Distributions", "Correlations", "Target"])
    
    with tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing", int(df.isnull().sum().sum()))
        col4.metric("Duplicates", df.duplicated().sum())
        st.dataframe(df.describe().round(3), use_container_width=True)
    
    with tabs[1]:
        feat = st.selectbox("Feature:", num_cols)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram", "Box Plot"])
        fig.add_trace(go.Histogram(x=df[feat], marker_color='#3b82f6', opacity=0.8), row=1, col=1)
        fig.add_trace(go.Box(y=df[feat], marker_color='#22c55e'), row=1, col=2)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, color_continuous_scale='RdBu', zmin=-1, zmax=1, text_auto='.2f')
        fig.update_layout(title="Correlation Matrix", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        feat_x = st.selectbox("X-axis:", [f for f in num_cols if f != target])
        fig = px.scatter(df, x=feat_x, y=target, trendline='ols', color_discrete_sequence=['#3b82f6'])
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Continue →"):
        next_step()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Data Engineering
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    df = st.session_state.df_clean.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    st.subheader("Data Engineering")
    
    # Missing Values
    missing_cols = [c for c in num_cols if df[c].isnull().any()]
    if missing_cols:
        st.markdown("**Missing Value Imputation**")
        method = st.selectbox("Method:", ["Mean", "Median", "Mode"])
        if st.button("Apply Imputation"):
            for col in missing_cols:
                if method == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.session_state.df_clean = df
            st.success(f"Imputed {len(missing_cols)} columns")
    else:
        st.success("No missing values")
    
    st.markdown("---")
    
    # Outliers
    st.markdown("**Outlier Detection**")
    outlier_method = st.selectbox("Method:", ["IQR", "Isolation Forest"])
    feat_outlier = st.multiselect("Features:", num_cols, default=num_cols[:3])
    
    if feat_outlier and st.button("Detect Outliers"):
        X_out = df[feat_outlier].dropna()
        
        if outlier_method == "IQR":
            mask = pd.Series([False] * len(X_out), index=X_out.index)
            for col in feat_outlier:
                Q1, Q3 = X_out[col].quantile(0.25), X_out[col].quantile(0.75)
                IQR = Q3 - Q1
                mask |= (X_out[col] < Q1 - 1.5*IQR) | (X_out[col] > Q3 + 1.5*IQR)
            outlier_indices = X_out[mask].index.tolist()
        else:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(StandardScaler().fit_transform(X_out))
            outlier_indices = X_out.index[preds == -1].tolist()
        
        st.session_state.outlier_indices = outlier_indices
        st.warning(f"Found {len(outlier_indices)} outliers ({len(outlier_indices)/len(df)*100:.1f}%)")
    
    if st.session_state.outlier_indices:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Remove Outliers"):
                df = df.drop(index=st.session_state.outlier_indices, errors='ignore')
                st.session_state.df_clean = df
                st.session_state.outlier_indices = []
                st.rerun()
        with col2:
            if st.button("Keep Outliers"):
                st.session_state.outlier_indices = []
    
    if st.button("Continue →"):
        next_step()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Feature Selection
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    df = st.session_state.df_clean
    target = st.session_state.target
    features = [f for f in st.session_state.features if f in df.columns]
    num_feats = df[features].select_dtypes(include=np.number).columns.tolist()
    
    st.subheader("Feature Selection")
    
    method = st.selectbox("Method:", ["Variance Threshold", "Correlation Filter", "Mutual Information"])
    
    if method == "Variance Threshold":
        from sklearn.feature_selection import VarianceThreshold
        thresh = st.slider("Threshold:", 0.0, 5.0, 0.1)
        sel = VarianceThreshold(threshold=thresh)
        sel.fit(df[num_feats])
        selected = [f for f, s in zip(num_feats, sel.get_support()) if s]
    
    elif method == "Correlation Filter":
        thresh = st.slider("Max Correlation:", 0.5, 1.0, 0.9)
        corr = df[num_feats].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > thresh)]
        selected = [f for f in num_feats if f not in to_drop]
    
    else:
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        if st.session_state.problem_type == "Regression":
            scores = mutual_info_regression(df[num_feats].fillna(0), df[target].fillna(0))
        else:
            scores = mutual_info_classif(df[num_feats].fillna(0), df[target].fillna(0))
        mi = pd.Series(scores, index=num_feats)
        thresh = st.slider("Min MI:", 0.0, float(mi.max()), float(mi.median()))
        selected = mi[mi >= thresh].index.tolist()
    
    final_features = st.multiselect("Final features:", num_feats, default=selected)
    st.info(f"Selected {len(final_features)} features")
    
    if st.button("Continue →") and final_features:
        st.session_state.selected_features = final_features
        next_step()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Data Split
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    df = st.session_state.df_clean
    target = st.session_state.target
    features = st.session_state.selected_features
    
    st.subheader("Train/Test Split")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size %", 10, 40, 20) / 100
    with col2:
        seed = st.number_input("Random Seed", 0, 999, 42)
    with col3:
        scale = st.checkbox("Standardize", value=True)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    X = df[features].select_dtypes(include=np.number).fillna(0)
    y = df[target].fillna(0)
    
    if st.session_state.problem_type == "Classification" and y.dtype == 'object':
        y = pd.Series(LabelEncoder().fit_transform(y))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
        st.session_state.scaler = scaler
    
    metric_cards({
        "Train": f"{len(X_train):,}",
        "Test": f"{len(X_test):,}",
        "Features": str(len(features)),
    })
    
    if st.button("Continue →"):
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        next_step()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Model Selection
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 6:
    st.subheader("Model Selection")
    
    problem = st.session_state.problem_type
    
    if problem == "Regression":
        models = ["Linear Regression", "SVM Regressor", "Random Forest Regressor"]
    else:
        models = ["Logistic Regression", "SVM Classifier", "Random Forest Classifier"]
    
    model_name = st.selectbox("Select Model:", models)
    
    # Model params
    if "SVM" in model_name:
        col1, col2 = st.columns(2)
        with col1:
            kernel = st.selectbox("Kernel:", ["rbf", "linear", "poly"])
        with col2:
            C = st.slider("C:", 0.01, 100.0, 1.0)
        st.session_state['svm_kernel'] = kernel
        st.session_state['svm_C'] = C
    
    elif "Random Forest" in model_name:
        col1, col2 = st.columns(2)
        with col1:
            n_trees = st.slider("Trees:", 10, 500, 100)
        with col2:
            max_depth = st.slider("Max Depth:", 1, 30, 10)
        st.session_state['rf_trees'] = n_trees
        st.session_state['rf_depth'] = max_depth
    
    if st.button("Continue →"):
        st.session_state.model_name = model_name
        next_step()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Training
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 7:
    st.subheader("Model Training")
    
    k = st.slider("K-Fold CV:", 2, 10, 5)
    
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    model_name = st.session_state.model_name
    problem = st.session_state.problem_type
    
    if st.button("Train Model"):
        from sklearn.model_selection import KFold, cross_val_score
        
        # Build model
        if model_name == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
        elif "SVM" in model_name and problem == "Regression":
            from sklearn.svm import SVR
            model = SVR(kernel=st.session_state.get('svm_kernel', 'rbf'), C=st.session_state.get('svm_C', 1.0))
        elif "SVM" in model_name:
            from sklearn.svm import SVC
            model = SVC(kernel=st.session_state.get('svm_kernel', 'rbf'), C=st.session_state.get('svm_C', 1.0))
        elif "Random Forest" in model_name and problem == "Regression":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=st.session_state.get('rf_trees', 100),
                                          max_depth=st.session_state.get('rf_depth', 10), random_state=42)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=st.session_state.get('rf_trees', 100),
                                           max_depth=st.session_state.get('rf_depth', 10), random_state=42)
        
        scoring = 'r2' if problem == "Regression" else 'accuracy'
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring)
        model.fit(X_train, y_train)
        
        st.session_state.model = model
        st.session_state.results['cv_scores'] = cv_scores
        
        st.success(f"Trained! CV {scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        fig = go.Figure(go.Bar(x=[f"Fold {i+1}" for i in range(k)], y=cv_scores, marker_color='#3b82f6'))
        fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="#eab308")
        fig.update_layout(title="Cross-Validation Scores", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.model and st.button("Continue →"):
        next_step()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Metrics
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 8:
    st.subheader("Performance Metrics")
    
    model = st.session_state.model
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    problem = st.session_state.problem_type
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    if problem == "Regression":
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        
        metric_cards({
            "Train R²": f"{train_r2:.4f}",
            "Test R²": f"{test_r2:.4f}",
            "RMSE": f"{test_rmse:.3f}",
            "MAE": f"{test_mae:.3f}",
        })
        
        gap = train_r2 - test_r2
        if gap > 0.1:
            st.error(f"Overfitting detected (gap: {gap:.3f})")
        elif test_r2 < 0.5:
            st.warning("Possible underfitting")
        else:
            st.success("Good fit")
        
        fig = px.scatter(x=y_test, y=test_preds, labels={'x': 'Actual', 'y': 'Predicted'},
                         trendline='ols', color_discrete_sequence=['#3b82f6'])
        fig.update_layout(title="Actual vs Predicted", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        from sklearn.metrics import accuracy_score, confusion_matrix
        
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        
        metric_cards({
            "Train Accuracy": f"{train_acc:.4f}",
            "Test Accuracy": f"{test_acc:.4f}",
            "Gap": f"{abs(train_acc - test_acc):.4f}",
        })
        
        cm = confusion_matrix(y_test, test_preds)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Continue →"):
        next_step()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: Tuning
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 9:
    st.subheader("Hyperparameter Tuning")
    
    model_name = st.session_state.model_name
    problem = st.session_state.problem_type
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    search_method = st.selectbox("Search:", ["Grid Search", "Random Search"])
    
    # Param grids
    if "Linear" in model_name or "Logistic" in model_name:
        param_grid = {'fit_intercept': [True, False]}
    elif "SVM" in model_name:
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    else:
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
    
    st.write("**Parameters:**", param_grid)
    cv_k = st.slider("CV Folds:", 2, 10, 3)
    
    if st.button("Start Tuning"):
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        
        # Rebuild base model
        if model_name == "Linear Regression":
            from sklearn.linear_model import LinearRegression; base = LinearRegression()
        elif model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression; base = LogisticRegression(max_iter=1000)
        elif "SVM" in model_name and problem == "Regression":
            from sklearn.svm import SVR; base = SVR()
        elif "SVM" in model_name:
            from sklearn.svm import SVC; base = SVC()
        elif "Random Forest" in model_name and problem == "Regression":
            from sklearn.ensemble import RandomForestRegressor; base = RandomForestRegressor(random_state=42)
        else:
            from sklearn.ensemble import RandomForestClassifier; base = RandomForestClassifier(random_state=42)
        
        scoring = 'r2' if problem == "Regression" else 'accuracy'
        
        with st.spinner("Tuning..."):
            if search_method == "Grid Search":
                searcher = GridSearchCV(base, param_grid, cv=cv_k, scoring=scoring, n_jobs=-1)
            else:
                searcher = RandomizedSearchCV(base, param_grid, cv=cv_k, n_iter=10, scoring=scoring, n_jobs=-1)
            searcher.fit(X_train, y_train)
        
        st.success(f"Best Score: {searcher.best_score_:.4f}")
        st.write("**Best Params:**", searcher.best_params_)
        
        # Compare
        old_preds = st.session_state.model.predict(X_test)
        new_preds = searcher.best_estimator_.predict(X_test)
        
        if problem == "Regression":
            from sklearn.metrics import r2_score
            old_score = r2_score(y_test, old_preds)
            new_score = r2_score(y_test, new_preds)
        else:
            from sklearn.metrics import accuracy_score
            old_score = accuracy_score(y_test, old_preds)
            new_score = accuracy_score(y_test, new_preds)
        
        fig = go.Figure(go.Bar(x=["Before", "After"], y=[old_score, new_score],
                               marker_color=['#737373', '#22c55e'],
                               text=[f"{old_score:.4f}", f"{new_score:.4f}"], textposition='outside'))
        fig.update_layout(title="Before vs After Tuning", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        
        st.session_state.model = searcher.best_estimator_
    
    st.markdown("---")
    st.success("🎉 Pipeline Complete!")
    
    if st.button("Start New Pipeline"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
