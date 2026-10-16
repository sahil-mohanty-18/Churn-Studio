SELECT payment_method AS segment,COUNT(*) AS customers,AVG(churn)*100 AS churn_rate FROM customers GROUP BY payment_method;
