import express from "express";
import marcadores from "./localizacion.js";

const app = express();
const port = 3000;

app.use(express.json());

app.get("/", (_, res) => {
  res.send("SpoTICfy API working!");
});

// La ruta y los marcadores los resuelve la raspi con data/grafo.json; acá queda solo la voz.
app.get("/destinos/buscar", marcadores.buscardestino);

const server = app.listen(port, () => {
  console.log(`SpoTICfy API listening at http://localhost:${port}`);
});

export { app, server };
