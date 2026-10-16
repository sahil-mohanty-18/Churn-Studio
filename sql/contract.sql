SELECT contract AS segment,COUNT(*) AS customers,AVG(churn)*100 AS churn_rate FROM customers GROUP BY contract;
