
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="HR Executive Intelligence Dashboard : By POWELL NDLOVU",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hr_attrition.csv"
REPORT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "HR_Executive_Summary_to_CEO.pdf"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
    df["AgeBand"] = pd.cut(df["Age"], bins=[17,25,35,45,55,65], labels=["18-25","26-35","36-45","46-55","56-65"])
    df["TenureBand"] = pd.cut(df["YearsAtCompany"], bins=[-1,1,3,5,10,20,40], labels=["0-1","2-3","4-5","6-10","11-20","21+"])
    df["IncomeBand"] = pd.qcut(df["MonthlyIncome"], 4, labels=["Lower","Mid-Lower","Mid-Upper","Upper"], duplicates="drop")
    return df

@st.cache_data
def score_attrition_risk(df):
    model_df = df.copy()

    candidate_numeric = [
        "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
        "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
        "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
        "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]
    numeric_features = [c for c in candidate_numeric if c in model_df.columns]

    candidate_categorical = [
        "BusinessTravel", "Department", "EducationField", "Gender", "JobRole",
        "MaritalStatus", "OverTime"
    ]
    categorical_features = [c for c in candidate_categorical if c in model_df.columns]

    X = model_df[numeric_features + categorical_features]
    y = model_df["AttritionFlag"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    scored = model_df.copy()
    scored["PredictedAttritionRisk"] = probs * 100
    scored["RiskBand"] = pd.cut(
        scored["PredictedAttritionRisk"],
        bins=[-0.01, 20, 40, 60, 100],
        labels=["Low", "Moderate", "High", "Critical"]
    )

    feature_names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "AbsCoefficient": np.abs(coefs)
    }).sort_values("AbsCoefficient", ascending=False)

    return scored, importance

def add_styles():
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(180deg, #f7f9fc 0%, #eef2f6 100%);}
    .block-container {padding-top: 1.1rem; padding-bottom: 1rem;}
    .hero-card {
        background: linear-gradient(135deg, #1f3c88 0%, #3b82f6 100%);
        color: white; padding: 1.25rem 1.35rem; border-radius: 20px;
        box-shadow: 0 8px 24px rgba(31,60,136,0.18); margin-bottom: 1rem;
    }
    .insight-box {
        background: white; border: 1px solid #d7e2eb; border-radius: 18px;
        padding: 1rem 1.1rem; margin-bottom: .8rem;
        box-shadow: 0 6px 16px rgba(0,0,0,0.05);
    }
    .section-title {font-size: 1.05rem; font-weight: 700; color: #243b53; margin-bottom: .35rem;}
    .small-note {color: #627d98; font-size: .90rem; line-height: 1.55;}
    .caption {
        color: #52606d; font-size: .87rem; line-height: 1.55;
        margin-top: .35rem; margin-bottom: .8rem;
    }
    </style>
    """, unsafe_allow_html=True)

def pct(v):
    return f"{v:.1f}%"

def filtered_data(df):
    st.sidebar.markdown("## Filters")
    dept = st.sidebar.multiselect("Department", sorted(df["Department"].unique()), default=sorted(df["Department"].unique()))
    role = st.sidebar.multiselect("Job Role", sorted(df["JobRole"].unique()), default=sorted(df["JobRole"].unique()))
    gender = st.sidebar.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
    overtime = st.sidebar.multiselect("OverTime", sorted(df["OverTime"].unique()), default=sorted(df["OverTime"].unique()))
    travel = st.sidebar.multiselect("Business Travel", sorted(df["BusinessTravel"].unique()), default=sorted(df["BusinessTravel"].unique()))
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
    out = df[
        df["Department"].isin(dept)
        & df["JobRole"].isin(role)
        & df["Gender"].isin(gender)
        & df["OverTime"].isin(overtime)
        & df["BusinessTravel"].isin(travel)
        & df["Age"].between(age_range[0], age_range[1])
    ].copy()
    return out

def executive_overview(df):
    overall_rate = df["AttritionFlag"].mean() * 100
    headcount = len(df)
    attritions = int(df["AttritionFlag"].sum())
    avg_income = df["MonthlyIncome"].mean()
    avg_tenure = df["YearsAtCompany"].mean()

    role_risk = df.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values(ascending=False)
    dept_risk = df.groupby("Department")["AttritionFlag"].mean().mul(100).sort_values(ascending=False)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Headcount", f"{headcount:,}")
    c2.metric("Attrition Rate", pct(overall_rate))
    c3.metric("Observed Exits", f"{attritions:,}")
    c4.metric("Avg Monthly Income", f"${avg_income:,.0f}")
    c5.metric("Avg Tenure", f"{avg_tenure:.1f} yrs")

    col1, col2 = st.columns((1.2,1))
    with col1:
        role_df = role_risk.reset_index(name="AttritionRate")
        fig = px.bar(role_df, x="JobRole", y="AttritionRate", color="AttritionRate",
                     title="Attrition Rate by Job Role", text_auto=".1f")
        fig.update_layout(height=420, xaxis_title="", yaxis_title="Attrition Rate (%)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="caption">Roles are not equally exposed. Attrition is concentrated in a few frontline and technical roles, which makes blanket retention programmes less efficient than targeted interventions.</div>', unsafe_allow_html=True)

    with col2:
        dept_df = dept_risk.reset_index(name="AttritionRate")
        fig2 = px.bar(dept_df, x="Department", y="AttritionRate", color="Department",
                      title="Attrition Rate by Department", text_auto=".1f")
        fig2.update_layout(height=420, xaxis_title="", yaxis_title="Attrition Rate (%)", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="caption">Departmental risk varies meaningfully. This helps leadership focus retention effort where workforce instability is materially above average.</div>', unsafe_allow_html=True)

    top_role = role_risk.index[0] if len(role_risk) else "n/a"
    top_dept = dept_risk.index[0] if len(dept_risk) else "n/a"
    st.markdown(f"""
    <div class="insight-box">
      <div class="section-title">Executive read-out</div>
      <div class="small-note">
      This filtered workforce view covers <b>{headcount:,}</b> employees with an attrition rate of <b>{overall_rate:.1f}%</b>.
      The highest-risk job role is <b>{top_role}</b>, while <b>{top_dept}</b> is the most exposed department.
      The pattern suggests attrition is concentrated in identifiable pressure points rather than spread evenly across the organisation.
      </div>
    </div>
    """, unsafe_allow_html=True)

def attrition_drivers(df):
    st.markdown("### Attrition Drivers")
    col1, col2 = st.columns(2)

    with col1:
        overtime_df = df.groupby("OverTime", as_index=False)["AttritionFlag"].mean()
        overtime_df["AttritionRate"] = overtime_df["AttritionFlag"] * 100
        fig = px.bar(overtime_df, x="OverTime", y="AttritionRate", color="OverTime",
                     title="Attrition Rate by Overtime", text_auto=".1f")
        fig.update_layout(height=360, xaxis_title="", yaxis_title="Attrition Rate (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="caption">Employees working overtime leave at a much higher rate. This is a workload signal, not just a scheduling detail.</div>', unsafe_allow_html=True)

    with col2:
        travel_df = df.groupby("BusinessTravel", as_index=False)["AttritionFlag"].mean()
        travel_df["AttritionRate"] = travel_df["AttritionFlag"] * 100
        travel_df = travel_df.sort_values("AttritionRate", ascending=False)
        fig2 = px.bar(travel_df, x="BusinessTravel", y="AttritionRate", color="BusinessTravel",
                      title="Attrition Rate by Business Travel", text_auto=".1f")
        fig2.update_layout(height=360, xaxis_title="", yaxis_title="Attrition Rate (%)", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="caption">Travel intensity is linked to higher attrition. Frequent travel appears to add measurable strain in the employee experience.</div>', unsafe_allow_html=True)

    driver = pd.DataFrame({
        "Factor": ["OverTime: Yes", "Travel Frequently", "Company Average"],
        "AttritionRate": [
            df.loc[df["OverTime"] == "Yes", "AttritionFlag"].mean() * 100 if (df["OverTime"] == "Yes").any() else np.nan,
            df.loc[df["BusinessTravel"] == "Travel_Frequently", "AttritionFlag"].mean() * 100 if (df["BusinessTravel"] == "Travel_Frequently").any() else np.nan,
            df["AttritionFlag"].mean() * 100
        ]
    }).dropna()

    fig3 = px.bar(driver, x="Factor", y="AttritionRate", color="Factor",
                  title="Key Risk Triggers vs Company Baseline", text_auto=".1f")
    fig3.update_layout(height=380, xaxis_title="", yaxis_title="Attrition Rate (%)", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="caption">This comparison shows which risk triggers rise materially above baseline. It helps leadership distinguish real pressure points from normal variation.</div>', unsafe_allow_html=True)

def workforce_segments(df):
    st.markdown("### Workforce Risk Segments")
    col1, col2 = st.columns(2)

    with col1:
        age_df = df.groupby("AgeBand", observed=False, as_index=False)["AttritionFlag"].mean()
        age_df["AttritionRate"] = age_df["AttritionFlag"] * 100
        fig = px.line(age_df, x="AgeBand", y="AttritionRate", markers=True, title="Attrition Rate by Age Band")
        fig.update_layout(height=360, xaxis_title="", yaxis_title="Attrition Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="caption">Younger employees are visibly more likely to leave. That points to early-career retention as a strategic HR priority.</div>', unsafe_allow_html=True)

    with col2:
        tenure_df = df.groupby("TenureBand", observed=False, as_index=False)["AttritionFlag"].mean()
        tenure_df["AttritionRate"] = tenure_df["AttritionFlag"] * 100
        fig2 = px.line(tenure_df, x="TenureBand", y="AttritionRate", markers=True, title="Attrition Rate by Years at Company")
        fig2.update_layout(height=360, xaxis_title="", yaxis_title="Attrition Rate (%)")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="caption">The first year is the steepest attrition zone. If leadership wants quick impact, this is the most obvious window to strengthen onboarding and manager support.</div>', unsafe_allow_html=True)

    role_df = df.groupby("JobRole", as_index=False).agg(
        Headcount=("AttritionFlag","size"),
        AttritionRate=("AttritionFlag","mean")
    )
    role_df["AttritionRate"] *= 100
    role_df["RiskSegment"] = np.where(
        (role_df["AttritionRate"] >= role_df["AttritionRate"].median()) & (role_df["Headcount"] >= role_df["Headcount"].median()),
        "High Risk / High Impact",
        np.where(
            (role_df["AttritionRate"] >= role_df["AttritionRate"].median()) & (role_df["Headcount"] < role_df["Headcount"].median()),
            "High Risk / Lower Scale",
            np.where(
                (role_df["AttritionRate"] < role_df["AttritionRate"].median()) & (role_df["Headcount"] >= role_df["Headcount"].median()),
                "Lower Risk / High Scale",
                "Lower Risk / Lower Scale"
            )
        )
    )

    fig3 = px.scatter(role_df, x="Headcount", y="AttritionRate", color="RiskSegment", size="Headcount",
                      text="JobRole", title="Job Role Opportunity Matrix")
    fig3.update_traces(textposition="top center")
    fig3.update_layout(height=450, xaxis_title="Role Headcount", yaxis_title="Attrition Rate (%)")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="caption">This matrix separates roles that are both high-risk and high-impact from roles that are only one or the other. It is useful for prioritising where HR action should start.</div>', unsafe_allow_html=True)

def predictive_risk_scoring(df, scored_full, importance):
    st.markdown("### Predictive Attrition Risk Scoring")
    filtered_scored = scored_full.loc[df.index].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Predicted Risk", pct(filtered_scored["PredictedAttritionRisk"].mean()))
    c2.metric("Critical-Risk Employees", int((filtered_scored["RiskBand"] == "Critical").sum()))
    c3.metric("High + Critical Share", pct((filtered_scored["RiskBand"].isin(["High", "Critical"]).mean() * 100)))

    col1, col2 = st.columns((1.15, 1))
    with col1:
        risk_counts = filtered_scored["RiskBand"].value_counts(dropna=False).rename_axis("RiskBand").reset_index(name="Employees")
        fig = px.bar(risk_counts, x="RiskBand", y="Employees", color="RiskBand",
                     title="Predicted Attrition Risk Distribution", text_auto=True,
                     category_orders={"RiskBand": ["Low", "Moderate", "High", "Critical"]})
        fig.update_layout(height=360, xaxis_title="", yaxis_title="Employees")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="caption">This model-based view estimates which employees are more likely to leave relative to others in the dataset. It supports prioritisation, not automatic decision-making.</div>', unsafe_allow_html=True)

    with col2:
        role_risk = filtered_scored.groupby("JobRole", as_index=False).agg(
            AvgPredictedRisk=("PredictedAttritionRisk", "mean"),
            Employees=("AttritionFlag", "size")
        ).sort_values("AvgPredictedRisk", ascending=False)
        fig2 = px.bar(role_risk, x="JobRole", y="AvgPredictedRisk", color="AvgPredictedRisk",
                      title="Average Predicted Risk by Job Role", text_auto=".1f")
        fig2.update_layout(height=360, xaxis_title="", yaxis_title="Predicted Risk (%)", coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="caption">Roles with elevated average predicted risk should be treated as priority watchlists for manager review, workload assessment, and retention planning.</div>', unsafe_allow_html=True)

    drivers = importance.head(10).copy()
    drivers["Direction"] = np.where(drivers["Coefficient"] > 0, "Raises Risk", "Lowers Risk")
    drivers["Feature"] = drivers["Feature"].str.replace("cat__", "", regex=False).str.replace("num__", "", regex=False)
    fig3 = px.bar(drivers.sort_values("AbsCoefficient"), x="AbsCoefficient", y="Feature", color="Direction",
                  orientation="h", title="Top Model Drivers of Attrition Risk")
    fig3.update_layout(height=420, xaxis_title="Coefficient Magnitude", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="caption">These are the strongest statistical drivers in the model. Use them as signals for investigation, not as proof of cause.</div>', unsafe_allow_html=True)

    preview_cols = ["EmployeeNumber", "Department", "JobRole", "OverTime", "BusinessTravel", "MonthlyIncome", "YearsAtCompany", "PredictedAttritionRisk", "RiskBand"]
    available_cols = [c for c in preview_cols if c in filtered_scored.columns]
    st.dataframe(filtered_scored[available_cols].sort_values("PredictedAttritionRisk", ascending=False).head(15), use_container_width=True)

def cost_of_attrition(df, scored_full):
    st.markdown("### Cost of Attrition by Role")
    filtered_scored = scored_full.loc[df.index].copy()

    c1, c2 = st.columns((1, 1.2))
    with c1:
        replacement_factor = st.slider(
            "Replacement cost multiplier (times annual salary)",
            min_value=0.25, max_value=2.00, value=0.50, step=0.05
        )
        st.markdown(
            '<div class="caption">This uses a simple planning assumption: cost of attrition = annual salary × replacement multiplier. Adjust the multiplier to reflect your organisation’s hiring, onboarding, and productivity-loss reality.</div>',
            unsafe_allow_html=True
        )

    filtered_scored["EstimatedAttritionCost"] = (filtered_scored["MonthlyIncome"] * 12 * replacement_factor) * (filtered_scored["PredictedAttritionRisk"] / 100)

    role_cost = filtered_scored.groupby("JobRole", as_index=False).agg(
        Employees=("AttritionFlag", "size"),
        ActualAttritionRate=("AttritionFlag", "mean"),
        AvgPredictedRisk=("PredictedAttritionRisk", "mean"),
        EstimatedCost=("EstimatedAttritionCost", "sum"),
        AvgMonthlyIncome=("MonthlyIncome", "mean")
    )
    role_cost["ActualAttritionRate"] *= 100
    role_cost = role_cost.sort_values("EstimatedCost", ascending=False)

    fig = px.bar(role_cost.head(10), x="JobRole", y="EstimatedCost", color="EstimatedCost",
                 title="Top 10 Job Roles by Estimated Cost of Attrition", text_auto=".2s")
    fig.update_layout(height=400, xaxis_title="", yaxis_title="Estimated Attrition Cost", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="caption">This combines exposure, compensation, and predicted risk to estimate where attrition is likely to be most financially painful.</div>', unsafe_allow_html=True)

    total_cost = role_cost["EstimatedCost"].sum()
    top_role = role_cost.iloc[0]["JobRole"] if len(role_cost) else "n/a"
    top_share = (role_cost.iloc[0]["EstimatedCost"] / total_cost * 100) if len(role_cost) and total_cost else np.nan

    st.markdown(f"""
    <div class="insight-box">
      <div class="section-title">Cost interpretation</div>
      <div class="small-note">
      Under the current planning assumption, the filtered workforce carries an estimated attrition cost exposure of <b>${total_cost:,.0f}</b>.
      <b>{top_role}</b> is the largest single cost hotspot, accounting for about <b>{top_share:.1f}%</b> of that exposure.
      This helps finance and HR prioritise retention where turnover is likely to hurt most.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(role_cost, use_container_width=True)

def action_tracking(df, scored_full):
    st.markdown("### Department Action Tracker")
    filtered_scored = scored_full.loc[df.index].copy()

    dept_actions = filtered_scored.groupby("Department", as_index=False).agg(
        Headcount=("AttritionFlag", "size"),
        ActualAttritionRate=("AttritionFlag", "mean"),
        AvgPredictedRisk=("PredictedAttritionRisk", "mean"),
        CriticalEmployees=("RiskBand", lambda s: int((s == "Critical").sum()))
    )
    dept_actions["ActualAttritionRate"] *= 100
    dept_actions["Priority"] = np.select(
        [
            (dept_actions["AvgPredictedRisk"] >= dept_actions["AvgPredictedRisk"].median()) & (dept_actions["CriticalEmployees"] >= dept_actions["CriticalEmployees"].median()),
            (dept_actions["AvgPredictedRisk"] >= dept_actions["AvgPredictedRisk"].median())
        ],
        ["Immediate", "High"],
        default="Monitor"
    )
    dept_actions["Recommended Action"] = np.select(
        [
            dept_actions["Priority"] == "Immediate",
            dept_actions["Priority"] == "High"
        ],
        [
            "Manager review, overtime check, stay interviews",
            "Targeted retention review and workload assessment"
        ],
        default="Track and reassess next cycle"
    )
    dept_actions["Owner"] = "HRBP + Department Head"
    dept_actions["Status"] = "Not started"
    dept_actions["Target Date"] = ""

    fig = px.scatter(
        dept_actions, x="Headcount", y="AvgPredictedRisk", size="CriticalEmployees",
        color="Priority", text="Department", title="Department Action Matrix"
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=400, xaxis_title="Department Headcount", yaxis_title="Average Predicted Risk (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="caption">Departments in the upper-right zone combine scale and elevated risk. That is where leadership attention is likely to pay off fastest.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
      <div class="section-title">Action tracking note</div>
      <div class="small-note">
      This dataset does not include a manager identifier, so true manager-level tracking cannot be built responsibly from the source file.
      The tracker below is therefore set at <b>department level</b>. If you add a manager field later, this same framework can be extended immediately.
      </div>
    </div>
    """, unsafe_allow_html=True)

    edited = st.data_editor(
        dept_actions[["Department", "Headcount", "ActualAttritionRate", "AvgPredictedRisk", "CriticalEmployees", "Priority", "Recommended Action", "Owner", "Status", "Target Date"]],
        use_container_width=True,
        num_rows="fixed",
        hide_index=True
    )

    action_csv = edited.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download department action tracker CSV",
        data=action_csv,
        file_name="department_action_tracker.csv",
        mime="text/csv",
        use_container_width=True
    )

def comp_and_experience(df):
    st.markdown("### Compensation, Satisfaction & Experience")
    col1, col2 = st.columns(2)

    with col1:
        income_df = df.groupby("IncomeBand", observed=False, as_index=False)["AttritionFlag"].mean()
        income_df["AttritionRate"] = income_df["AttritionFlag"] * 100
        fig = px.bar(income_df, x="IncomeBand", y="AttritionRate", color="IncomeBand",
                     title="Attrition Rate by Income Band", text_auto=".1f")
        fig.update_layout(height=360, xaxis_title="", yaxis_title="Attrition Rate (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="caption">Attrition pressure is typically strongest in lower income bands. This does not automatically mean compensation is the only issue, but it signals where reward design deserves scrutiny.</div>', unsafe_allow_html=True)

    with col2:
        sat_df = df.groupby("JobSatisfaction", as_index=False)["AttritionFlag"].mean()
        sat_df["AttritionRate"] = sat_df["AttritionFlag"] * 100
        fig2 = px.bar(sat_df, x="JobSatisfaction", y="AttritionRate", color="JobSatisfaction",
                      title="Attrition Rate by Job Satisfaction", text_auto=".1f")
        fig2.update_layout(height=360, xaxis_title="Job Satisfaction Level", yaxis_title="Attrition Rate (%)", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="caption">Lower job satisfaction aligns with higher attrition. This strengthens the case for manager quality, role design, and recognition interventions alongside pay discussions.</div>', unsafe_allow_html=True)

    summary = df.groupby("Department", as_index=False).agg(
        Headcount=("AttritionFlag","size"),
        AttritionRate=("AttritionFlag","mean"),
        AvgIncome=("MonthlyIncome","mean"),
        AvgYearsAtCompany=("YearsAtCompany","mean")
    )
    summary["AttritionRate"] *= 100
    st.dataframe(summary.sort_values("AttritionRate", ascending=False), use_container_width=True)
    st.markdown('<div class="caption">The table gives decision-makers a practical comparison view across departments: workforce size, attrition pressure, income level, and tenure profile in one place.</div>', unsafe_allow_html=True)

def strategic_takeaways(df):
    overall = df["AttritionFlag"].mean() * 100
    overtime_yes = df.loc[df["OverTime"]=="Yes","AttritionFlag"].mean() * 100 if (df["OverTime"]=="Yes").any() else np.nan
    early = df.loc[df["TenureBand"]=="0-1","AttritionFlag"].mean() * 100 if (df["TenureBand"]=="0-1").any() else np.nan
    top_role = df.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values(ascending=False)
    role_name = top_role.index[0] if len(top_role) else "n/a"
    role_rate = top_role.iloc[0] if len(top_role) else np.nan

    st.markdown("### Strategic Takeaways")
    take1, take2, take3 = st.columns(3)
    take1.markdown(f"""
    <div class="insight-box">
      <div class="section-title">1. Fix first-year retention</div>
      <div class="small-note">
      The first 12 months show the sharpest attrition pressure at <b>{early:.1f}%</b>, well above the baseline of <b>{overall:.1f}%</b>.
      Early-career onboarding and manager support should be treated as a priority intervention zone.
      </div>
    </div>
    """, unsafe_allow_html=True)
    take2.markdown(f"""
    <div class="insight-box">
      <div class="section-title">2. Treat overtime as a risk trigger</div>
      <div class="small-note">
      Employees working overtime show attrition around <b>{overtime_yes:.1f}%</b>, which is materially above the overall workforce rate.
      Retention and workload design need to be considered together.
      </div>
    </div>
    """, unsafe_allow_html=True)
    take3.markdown(f"""
    <div class="insight-box">
      <div class="section-title">3. Focus on exposed roles first</div>
      <div class="small-note">
      <b>{role_name}</b> is the most exposed role in the current view at <b>{role_rate:.1f}%</b>.
      Targeted action in the highest-risk roles will likely outperform broad, uniform HR programmes.
      </div>
    </div>
    """, unsafe_allow_html=True)

def executive_report(df):
    st.markdown("### Executive PDF Summary to the CEO")
    col1, col2 = st.columns((1.2, 1))
    with col1:
        st.markdown("""
        <div class="insight-box">
          <div class="section-title">Board-ready summary</div>
          <div class="small-note">
          Download the executive PDF summary for a concise CEO-facing brief on attrition risk, workforce pressure points,
          and priority HR actions. It is designed for stakeholder circulation and leadership discussion.
          </div>
        </div>
        """, unsafe_allow_html=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered HR data as CSV", data=csv, file_name="filtered_hr_attrition_data.csv", mime="text/csv", use_container_width=True)

    with col2:
        if REPORT_PATH.exists():
            with open(REPORT_PATH, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                "Download Executive PDF Summary",
                data=pdf_bytes,
                file_name="HR_Executive_Summary_to_CEO.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.caption("Includes executive summary, risk snapshot, and CEO recommendations.")
        else:
            st.warning("Executive PDF not found in outputs folder.")

def main():
    add_styles()
    df = load_data()
    scored_full, importance = score_attrition_risk(df)
    filtered = filtered_data(df)

    st.markdown("""
    <div class="hero-card">
        <h2 style="margin:0;">HR Executive Intelligence Dashboard</h2>
        <p style="margin:0.35rem 0 0 0;">
        A decision-focused people analytics dashboard designed to highlight attrition risk, workforce pressure points, and practical retention priorities for leadership.
        <h5 style="margin:0;">Created by Powell A. Ndlovu : GIS and Data Analyst</h5>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if filtered.empty:
        st.warning("No employees match the current filters. Broaden the selection to continue.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Executive Overview", "Attrition Drivers", "Risk Segments", "Predictive Risk",
        "Attrition Cost", "Action Tracker", "Executive Report"
    ])

    with tab1:
        executive_overview(filtered)
    with tab2:
        attrition_drivers(filtered)
    with tab3:
        workforce_segments(filtered)
        comp_and_experience(filtered)
        strategic_takeaways(filtered)
    with tab4:
        predictive_risk_scoring(filtered, scored_full, importance)
    with tab5:
        cost_of_attrition(filtered, scored_full)
    with tab6:
        action_tracking(filtered, scored_full)
    with tab7:
        executive_report(filtered)

if __name__ == "__main__":
    main()
