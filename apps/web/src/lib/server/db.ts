import { Pool, type PoolClient, type QueryResultRow } from "pg";

const DEFAULT_DATABASE_URL =
  "postgres://llm_craft:llm_craft_dev@localhost:5432/llm_craft";

declare global {
  var llmCraftPgPool: Pool | undefined;
}

export function getPool(): Pool {
  globalThis.llmCraftPgPool ??= new Pool({
    connectionString: process.env.DATABASE_URL ?? DEFAULT_DATABASE_URL
  });

  return globalThis.llmCraftPgPool;
}

export async function query<T extends QueryResultRow>(
  sql: string,
  params: unknown[] = []
) {
  return getPool().query<T>(sql, params);
}

export async function transaction<T>(
  callback: (client: PoolClient) => Promise<T>
): Promise<T> {
  const client = await getPool().connect();

  try {
    await client.query("BEGIN");
    const result = await callback(client);
    await client.query("COMMIT");
    return result;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}
