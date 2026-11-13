import unittest

import numpy as np

from entrenar_decodificador import (
    NBITS,
    binarizar_predicciones_rgb,
    mapear_huellas_por_molecule_id,
)


class AdaptacionRGBTests(unittest.TestCase):
    def test_binarizacion_reproduce_clip_y_top_100(self):
        scores = np.linspace(-1.0, 2.0, NBITS, dtype=np.float32)[None, :]

        (bits,) = binarizar_predicciones_rgb(scores)
        referencia = np.clip(scores[0], 0.0, 1.0)
        esperados = np.sort(np.argpartition(-referencia, 100)[:100])

        np.testing.assert_array_equal(bits, esperados)
        self.assertEqual(len(bits), 100)
        self.assertEqual(bits.dtype, np.int32)

    def test_binarizacion_admite_toda_la_huella(self):
        scores = np.zeros((1, NBITS), dtype=np.float32)

        (bits,) = binarizar_predicciones_rgb(scores, top_k=NBITS)

        self.assertEqual(len(bits), NBITS)
        self.assertEqual(len(np.unique(bits)), NBITS)

    def test_alinea_por_id_sin_depender_del_orden(self):
        huellas = [np.array([2], dtype=np.int32), np.array([7], dtype=np.int32)]

        resultado = mapear_huellas_por_molecule_id(
            ["mol-b", "mol-a"], huellas, ["mol-a", "mol-b"]
        )

        np.testing.assert_array_equal(resultado["mol-a"], [7])
        np.testing.assert_array_equal(resultado["mol-b"], [2])

    def test_rechaza_ids_faltantes_y_duplicados(self):
        with self.assertRaisesRegex(ValueError, "faltan 1"):
            mapear_huellas_por_molecule_id(["mol-a"], [[1]], ["mol-b"])
        with self.assertRaisesRegex(ValueError, "duplicado"):
            mapear_huellas_por_molecule_id(
                ["mol-a", "mol-a"], [[1], [2]], ["mol-a"]
            )

    def test_rechaza_scores_invalidos(self):
        with self.assertRaisesRegex(ValueError, "forma"):
            binarizar_predicciones_rgb(np.zeros((1, NBITS - 1)))
        scores = np.zeros((1, NBITS), dtype=np.float32)
        scores[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN o infinito"):
            binarizar_predicciones_rgb(scores)


if __name__ == "__main__":
    unittest.main()
