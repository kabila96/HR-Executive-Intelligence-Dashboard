# HR Executive Intelligence Dashboard

A Streamlit dashboard built to turn HR attrition data into executive-ready insight.

## Purpose
This project helps decision-makers identify where attrition risk is concentrated, which workforce segments are most exposed, and where targeted HR action is likely to have the strongest impact.

## What the dashboard includes
- Executive overview with headline KPIs
- Attrition risk by role and department
- Overtime and travel as risk triggers
- Age and tenure attrition patterns
- Compensation and job-satisfaction signals
- Strategic takeaways for leadership
- Downloadable Executive PDF Summary to the CEO

## Project structure
```text
hr_attrition_streamlit_dashboard/
├── dashboard/
│   └── app.py
├── data/
│   └── hr_attrition.csv
├── outputs/
│   └── HR_Executive_Summary_to_CEO.pdf
├── README.md
└── requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```


## Advanced decision-support features
- Predictive attrition risk scoring using logistic regression
- Role-level estimated cost of attrition with adjustable replacement-cost assumptions
- Department action tracker with editable status, owner, and target date fields

## Important note
The source data does not include a manager identifier. The dashboard therefore implements department-level action tracking responsibly. If a manager field is added later, the tracker can be extended to manager level immediately.
