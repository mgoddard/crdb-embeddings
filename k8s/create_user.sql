CREATE ROLE embeduser LOGIN PASSWORD 'embedpasswd';
GRANT admin TO embeduser; -- "embeduser" can log into DB Console and also use defaultdb
SET CLUSTER SETTING sql.txn.read_committed_isolation.enabled = true;

