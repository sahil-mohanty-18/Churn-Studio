SELECT risk AS segment,COUNT(*) AS customers FROM latest_predictions GROUP BY risk;
