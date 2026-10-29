"""Write the Colab cells (same code, different CONFIG line)."""
src = open("onecell.py").read()
line = 'CONFIG = dict(name="w1", site="lm",'
assert line in src
cells = {
    # vision tower of Gemma-3 (SigLIP, 27 layers, width 1152); 26 = the SAE++ site, 13 = middle
    "cell_vision.py": src.replace(line, 'CONFIG = dict(name="w1_vision", site="vision",')
                         .replace('layers=[12, 20, 28],', 'layers=[13, 26],', 1),
}
for name, code in cells.items():
    assert code != src
    open(name, "w").write(code)
    print("wrote", name)
