-- ---------------------------------------------------------------------------
-- Database bootstrap. Executed automatically by the pgvector image on first
-- start (mounted into /docker-entrypoint-initdb.d).
--
-- NOTE: the vector column width MUST match the ACTIVE embedding provider's
-- output dimension:
--   * Ollama  nomic-embed-text        -> 768   (current setting)
--   * OpenAI  text-embedding-3-small  -> 1536
--   * Fake provider                   -> EMBEDDING_DIM (default 1536)
-- If you switch embedding provider/model, change the width below to match and
-- recreate the volume (`docker compose down -v`) so chunks are re-ingested.
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================================================
-- RAG document chunks
-- ===========================================================================
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source       TEXT        NOT NULL,       -- filename / document id
    source_type  TEXT        NOT NULL,       -- 'pdf' | 'markdown' | 'text'
    chunk_index  INTEGER     NOT NULL,
    content      TEXT        NOT NULL,
    metadata     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    embedding    vector(768),     -- nomic-embed-text (Ollama); see NOTE above
    -- Postgres full-text search vector powers the BM25-style lexical retriever.
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Lexical (BM25-style) index over the generated tsvector.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);

-- Approximate-nearest-neighbour index for cosine vector similarity.
-- HNSW gives good recall/latency; requires pgvector >= 0.5.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Avoid duplicate ingestion of the same chunk of the same source.
CREATE UNIQUE INDEX IF NOT EXISTS chunks_source_chunk_uidx
    ON chunks (source, chunk_index);

-- ===========================================================================
-- Sample business data for the NL->SQL agent
-- ===========================================================================
CREATE TABLE IF NOT EXISTS orders (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_name TEXT        NOT NULL,
    product       TEXT        NOT NULL,
    category      TEXT        NOT NULL,
    quantity      INTEGER     NOT NULL,
    unit_price    NUMERIC(10,2) NOT NULL,
    status        TEXT        NOT NULL,      -- 'pending' | 'shipped' | 'delivered' | 'cancelled' | 'refunded'
    region        TEXT        NOT NULL,      -- 'NA' | 'EMEA' | 'APAC' | 'LATAM'
    order_date    DATE        NOT NULL
);

-- Seed data (idempotent: only insert when table is empty).
INSERT INTO orders (customer_name, product, category, quantity, unit_price, status, region, order_date)
SELECT * FROM (VALUES
    ('Acme Corp',        'Widget Pro',        'hardware',  120, 19.99,  'delivered', 'NA',    DATE '2024-01-15'),
    ('Globex',           'Widget Pro',        'hardware',   40, 19.99,  'shipped',   'EMEA',  DATE '2024-02-03'),
    ('Initech',          'Analytics Seat',    'software',   10, 149.00, 'delivered', 'NA',    DATE '2024-02-11'),
    ('Umbrella Ltd',     'Support Gold',      'service',     1, 999.00, 'pending',   'EMEA',  DATE '2024-03-01'),
    ('Hooli',            'Widget Mini',       'hardware',  500,  4.50,  'delivered', 'APAC',  DATE '2024-03-09'),
    ('Stark Industries', 'Analytics Seat',    'software',   50, 149.00, 'delivered', 'NA',    DATE '2024-03-22'),
    ('Wayne Enterprises','Support Gold',      'service',     3, 999.00, 'shipped',   'NA',    DATE '2024-04-02'),
    ('Cyberdyne',        'Widget Pro',        'hardware',   75, 19.99,  'cancelled', 'APAC',  DATE '2024-04-15'),
    ('Soylent Corp',     'Widget Mini',       'hardware',  200,  4.50,  'delivered', 'LATAM', DATE '2024-04-27'),
    ('Tyrell Corp',      'Analytics Seat',    'software',   25, 149.00, 'refunded',  'EMEA',  DATE '2024-05-05'),
    ('Vandelay',         'Widget Pro',        'hardware',   60, 19.99,  'delivered', 'NA',    DATE '2024-05-18'),
    ('Gringotts',        'Support Gold',      'service',     2, 999.00, 'delivered', 'EMEA',  DATE '2024-05-30'),
    ('Pied Piper',       'Analytics Seat',    'software',  100, 149.00, 'shipped',   'NA',    DATE '2024-06-08'),
    ('Aperture Science', 'Widget Mini',       'hardware',  350,  4.50,  'pending',   'APAC',  DATE '2024-06-19'),
    ('Massive Dynamic',  'Widget Pro',        'hardware',   90, 19.99,  'delivered', 'NA',    DATE '2024-06-25'),
    ('Wonka Industries', 'Support Gold',      'service',     5, 999.00, 'delivered', 'LATAM', DATE '2024-07-01'),
    ('Oscorp',           'Analytics Seat',    'software',   15, 149.00, 'cancelled', 'EMEA',  DATE '2024-07-11'),
    ('Nakatomi',         'Widget Pro',        'hardware',  110, 19.99,  'shipped',   'APAC',  DATE '2024-07-20'),
    ('Weyland-Yutani',   'Widget Mini',       'hardware',  420,  4.50,  'delivered', 'EMEA',  DATE '2024-08-02'),
    ('Monarch Sciences', 'Support Gold',      'service',     4, 999.00, 'pending',   'NA',    DATE '2024-08-14')
) AS seed
WHERE NOT EXISTS (SELECT 1 FROM orders);
