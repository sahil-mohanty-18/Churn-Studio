PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS customers (
 customer_id TEXT PRIMARY KEY, tenure INTEGER NOT NULL, contract TEXT NOT NULL,
 monthly_charges REAL NOT NULL, payment_method TEXT NOT NULL, churn INTEGER NOT NULL CHECK(churn IN (0,1)), profile TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_customers_contract ON customers(contract);
CREATE INDEX IF NOT EXISTS idx_customers_payment ON customers(payment_method);
CREATE TABLE IF NOT EXISTS model_runs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 model_name TEXT NOT NULL, metrics TEXT NOT NULL, source TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS predictions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT REFERENCES customers(customer_id) ON DELETE CASCADE,
 model_run_id INTEGER NOT NULL REFERENCES model_runs(id), probability REAL NOT NULL CHECK(probability BETWEEN 0 AND 1),
 risk TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_predictions_customer ON predictions(customer_id,id DESC);
CREATE VIEW IF NOT EXISTS latest_predictions AS SELECT p.* FROM predictions p WHERE p.id=(SELECT MAX(q.id) FROM predictions q WHERE q.customer_id=p.customer_id);
CREATE VIEW IF NOT EXISTS cohorts AS SELECT CASE WHEN tenure<12 THEN '00–11 months' WHEN tenure<36 THEN '12–35 months' ELSE '36+ months' END AS segment, COUNT(*) AS customers, AVG(churn)*100 AS churn_rate FROM customers GROUP BY segment;
