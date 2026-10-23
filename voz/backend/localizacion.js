import { buscarDestino } from "./busquedaService.js";

/** GET /destinos/buscar?texto=quiero ir al aula doscientos tres */
const buscardestino = async (req, res) => {
  try {
    const { texto } = req.query;
    if (!texto) return res.status(400).json({ error: "Falta el parámetro texto" });

    const resultado = await buscarDestino(texto);   // ← el await es lo importante
    if (!resultado) {
      return res.status(404).json({
        error: "No entendí el destino",
        pregunta: "No te entendí. ¿A dónde querés ir?",
      });
    }
    res.json(resultado);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error interno" });
  }
};

export default { buscardestino };
