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

  let releaseError: unknown;

  try {
    await client.query("BEGIN");
    const result = await callback(client);
    await client.query("COMMIT");
    return result;
  } catch (error) {
    // Preserve the original error even if ROLLBACK itself fails (e.g. a broken
    // connection), and flag the connection as poisoned so the pool discards it.
    try {
      await client.query("ROLLBACK");
    } catch (rollbackError) {
      releaseError = rollbackError;
    }

    throw error;
  } finally {
    client.release(releaseError ? (releaseError as Error) : undefined);
  }
}
