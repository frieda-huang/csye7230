### Basic House Cleaning

Since HNSW supports incremental indexing, we don’t DROP the index programmatically. As a result, when testing performance, we need to manually DROP the index beforehand to avoid any slowdown due to accumulated data.

### Implemented Optimization

#### Reduce Index Build Time

-   Applied `CREATE INDEX CONCURRENTLY ...`
    -   _Outcome_: it reduces total time slightly, but not by large margin

### Cleaning

Before running any tests, run the following commands one by one

```
-- Drop index first
DROP INDEX IF EXISTS flattened_embedding_index;

-- Reset pg_stat_statements
SELECT
pg_stat_statements_reset ();

-- Truncate all tables and restart identity columns
TRUNCATE flattened_embedding RESTART IDENTITY CASCADE;

TRUNCATE embedding RESTART IDENTITY CASCADE;

TRUNCATE folder RESTART IDENTITY CASCADE;

TRUNCATE page RESTART IDENTITY CASCADE;

TRUNCATE file RESTART IDENTITY CASCADE;

TRUNCATE query RESTART IDENTITY CASCADE;

-- Analyze tables for updated statistics
ANALYZE;
```

### Relevant GitHub Issues

-   https://github.com/pgvector/pgvector/issues/543
-   https://github.com/pgvector/pgvector/issues/500
-   https://github.com/pgvector/pgvector/issues/299
