import importlib
import sys
import unittest
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

sys.path.insert(0, str(Path(__file__).parent))
evaluador = importlib.import_module("30_evaluar_generativo")


class EvaluadorGenerativoTests(unittest.TestCase):
    def test_mrr_novel_uses_tautomerized_truth_and_excludes_train_truth(self):
        chem = Chem
        tautomerizer = rdMolStandardize.TautomerEnumerator()
        train_ik = evaluador.canonical_ik14(
            "c1ccccc1", chem, tautomerizer
        )[0]
        train_direct_ik = evaluador.direct_ik14("c1ccccc1", chem)[0]
        truths = {
            "novel": {"inchikey14": "", "smiles": "CCO"},
            "seen": {"inchikey14": "", "smiles": "c1ccccc1"},
        }
        pred_rows = [
            {"molecule_id": "novel", "smiles": "CCO"},
            {"molecule_id": "seen", "smiles": "c1ccccc1"},
        ]

        result = evaluador.evaluar(
            pred_rows, truths, chem, tautomerizer,
            train_ik14={train_ik}, train_direct_ik14={train_direct_ik},
            n_bootstrap=100,
        )

        self.assertEqual(result["mrr25_novel"]["n"], 1)
        self.assertEqual(result["mrr25_novel"]["mrr25"], 1.0)
        self.assertEqual(result["mrr25_novel"]["n_verdades_en_train"], 1)
        self.assertEqual(result["mrr25_novel_tautomer"], result["mrr25_novel"])
        self.assertEqual(result["mrr25_novel_direct"]["n"], 1)
        self.assertEqual(result["mrr25_novel_direct"]["n_verdades_en_train"], 1)

    def test_novelty_metrics_can_use_independent_overlap_policies(self):
        chem = Chem
        tautomerizer = rdMolStandardize.TautomerEnumerator()
        truth_tautomer_ik = evaluador.canonical_ik14(
            "CCO", chem, tautomerizer
        )[0]
        truths = {"mol": {"inchikey14": truth_tautomer_ik, "smiles": "CCO"}}
        pred_rows = [{"molecule_id": "mol", "smiles": "CCO"}]

        result = evaluador.evaluar(
            pred_rows, truths, chem, tautomerizer,
            train_ik14={truth_tautomer_ik}, train_direct_ik14=set(),
            n_bootstrap=0,
        )

        self.assertEqual(result["mrr25_novel_tautomer"]["n"], 0)
        self.assertIsNone(result["mrr25_novel_tautomer"]["mrr25"])
        self.assertEqual(
            result["mrr25_novel_direct"]["n"],
            1,
        )
        self.assertEqual(result["mrr25_novel_direct"]["mrr25"], 1.0)


if __name__ == "__main__":
    unittest.main()
