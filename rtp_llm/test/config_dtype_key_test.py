import os
import re
from unittest import TestCase, main

from rtp_llm.utils.util import get_config_dtype

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_ORPHAN_FAMILIES = ("qwen.py", "qwen2_vl.py", "glm4_moe.py")
DIRECT_KEY_READ = re.compile(
    r"""(?:\.get\(\s*["'](?:torch_)?dtype["']"""
    r"""|\[\s*["'](?:torch_)?dtype["']\s*\]"""
    r"""|getattr\([^,]+,\s*["'](?:torch_)?dtype["'])"""
)


class ConfigDtypeKeyTest(TestCase):
    def test_new_dtype_key_is_read(self):
        self.assertEqual(get_config_dtype({"dtype": "bfloat16"}), "bfloat16")

    def test_legacy_torch_dtype_key_is_read(self):
        self.assertEqual(get_config_dtype({"torch_dtype": "bfloat16"}), "bfloat16")

    def test_new_key_wins_when_both_declared(self):
        both = {"dtype": "bfloat16", "torch_dtype": "float16"}
        self.assertEqual(get_config_dtype(both), "bfloat16")

    def test_legacy_key_used_when_new_key_is_null(self):
        self.assertEqual(
            get_config_dtype({"dtype": None, "torch_dtype": "float16"}), "float16"
        )

    def test_neither_key_declared_returns_none(self):
        self.assertIsNone(get_config_dtype({"hidden_size": 4096}))

    def test_non_string_value_is_not_a_declaration(self):
        # transformers also accepts per-module dtype dicts; passing one on would reach
        # WEIGHT_TYPE.from_str and raise AttributeError on a value that used to be ignored.
        self.assertIsNone(get_config_dtype({"dtype": {"default": "bfloat16"}}))
        self.assertIsNone(get_config_dtype({"torch_dtype": 16}))
        self.assertEqual(
            get_config_dtype({"dtype": {"default": "x"}, "torch_dtype": "bfloat16"}),
            "bfloat16",
        )

    def test_blank_value_does_not_shadow_the_other_key(self):
        self.assertIsNone(get_config_dtype({"dtype": "  "}))
        self.assertEqual(
            get_config_dtype({"dtype": "", "torch_dtype": "bfloat16"}), "bfloat16"
        )

    def test_absent_config_returns_none(self):
        self.assertIsNone(get_config_dtype(None))
        self.assertIsNone(get_config_dtype({}))

    def test_precedence_matches_transformers(self):
        # The precedence above is not a guess: pin it to the library that writes these
        # files, so a future rename lands as a red test instead of a silent dtype swap.
        from transformers import PretrainedConfig

        config = PretrainedConfig(dtype="bfloat16", torch_dtype="float16")
        declared = get_config_dtype({"dtype": "bfloat16", "torch_dtype": "float16"})
        self.assertEqual(str(config.dtype), "torch." + declared)

    def test_orphan_families_now_read_the_declared_dtype(self):
        # The three families this package fixes (qwen / qwen2_vl / glm4_moe) previously
        # parsed config.json without ever reading the declared dtype, so they silently
        # fell back to FP16. They must now route the read through get_config_dtype.
        # The repo-wide "no model reads a dtype key directly" scan lives in the follow-up
        # root-cause helper package; here we only assert the three orphan families were
        # converted (and do not read a dtype key directly).
        offenders = []
        checked = 0
        for name in _ORPHAN_FAMILIES:
            path = os.path.join(MODELS_DIR, name)
            if not os.path.exists(path):
                continue
            checked += 1
            with open(path, "r", encoding="utf-8") as reader:
                text = reader.read()
            self.assertIn(
                "get_config_dtype", text, "%s must route dtype through the helper" % name
            )
            for lineno, line in enumerate(text.splitlines(), start=1):
                if DIRECT_KEY_READ.search(line):
                    offenders.append("%s:%d %s" % (path, lineno, line.strip()))
        self.assertEqual(checked, len(_ORPHAN_FAMILIES), "orphan model sources not found")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    main()
