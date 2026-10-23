import pkg from "pg";
const { Pool } = pkg;

// Las credenciales van en voz/backend/.env (ver .env.example). Nunca en el código.
export const pool = new Pool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port: Number(process.env.DB_PORT ?? 5432),
  ssl: {
    rejectUnauthorized: false,
  },
  channelBinding: "require",
});

export const dbController = {
  pool: pool,
};

export const query = async (text, params = []) => {
  const client = await dbController.pool.connect();
  try {
    const result = await client.query(text, params);
    return result;
  } finally {
    client.release();
  }
};
