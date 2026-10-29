"""Write the Colab cells (same code, different CONFIG lines)."""
src = open("onecell.py").read()
head = 'CONFIG = dict(name="w1", site="lm", family="gemma3", pt="google/gemma-3-4b-pt", it="google/gemma-3-4b-it",'
lay = 'layers=[12, 20, 28], n_images=5000, tok_per_image=32, batch=16,'
insp = 'extra_seeds=[1, 2], inspect_layers=[20],'
for x in (head, lay, insp):
    assert src.count(x) == 1, x
cells = {
    # vision tower of Gemma-3 (SigLIP, 27 layers, width 1152); 26 = the SAE++ site, 13 = middle
    "cell_vision.py": src.replace(head, head.replace('name="w1", site="lm"', 'name="w1_vision", site="vision"'))
                         .replace(lay, lay.replace("[12, 20, 28]", "[13, 26]"))
                         .replace(insp, insp.replace("[20]", "[26]")),
    # second model pair: Qwen2-VL-2B base vs Instruct (28 LM layers, width 1536; 10/16/23 = same relative depth as
    # Gemma 12/20/28 of 34). Not gated. Smaller model -> batch 32.
    "cell_qwen2vl.py": src.replace(head, 'CONFIG = dict(name="w2_qwen2vl", site="lm", family="qwen2vl", '
                                         'pt="Qwen/Qwen2-VL-2B", it="Qwen/Qwen2-VL-2B-Instruct",')
                          .replace(lay, lay.replace("[12, 20, 28]", "[10, 16, 23]").replace("batch=16", "batch=32"))
                          .replace(insp, insp.replace("[20]", "[16]")),
}
for name, code in cells.items():
    assert code != src
    open(name, "w").write(code)
    print("wrote", name)
