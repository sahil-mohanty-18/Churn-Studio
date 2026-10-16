SELECT COUNT(*) AS total_customers, ROUND(AVG(c.churn)*100,2) AS churn_rate,
SUM(CASE WHEN p.risk='High' THEN 1 ELSE 0 END) AS customers_at_risk,
ROUND(SUM(c.monthly_charges*COALESCE(p.probability,0)),2) AS revenue_at_risk
FROM customers c LEFT JOIN latest_predictions p USING(customer_id);
