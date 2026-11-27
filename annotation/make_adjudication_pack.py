"""REVISION: blind adjudication of the crops on which the two annotators disagree (or annotator 2 could not tell).

A third person (e.g. the supervisor) sees each disputed crop with the TWO candidate labels as options A and B,
in random order, without knowing which annotator gave which label and without any model output, and picks
A, B, "Neither (other sign)" or "Cannot tell". Example pictures of both candidate classes are shown.

Usage (project folder):
    python make_adjudication_pack.py annotator_key.csv labels_annotator2.csv annotator_key_part2.csv labels_annotator2_part2.csv
Gives:  adjudication_pack/   (zip and send; open index.html, click, then "Download my choices")
        adjudication_key.csv (KEEP: which option is which annotator)
"""
import csv, json, random, shutil, sys
from pathlib import Path
args = sys.argv[1:]
key, lab = {}, {}
for kf, lf in zip(args[0::2], args[1::2]):
    for r in csv.DictReader(open(kf, encoding="utf-8")): key[r["item"]] = r
    for r in csv.DictReader(open(lf, encoding="utf-8")):
        if r["label"]: lab[r["item"]] = r["label"]
def norm(c): return "9999_other_sign" if c in ("9999_other_sign", "OTHER") else c
disp = [i for i in sorted(key) if i in lab and (lab[i] == "UNSURE" or norm(lab[i]) != norm(key[i]["true_class"]))]
classes = sorted(d.name for d in Path("ts1m_ma_48_clean/train").iterdir() if d.is_dir())
out = Path("adjudication_pack"); shutil.rmtree(out, ignore_errors=True); (out / "crops").mkdir(parents=True); (out / "classes").mkdir()
random.seed(77); random.shuffle(disp)
def ex(c):
    if c == "9999_other_sign": return ""
    f = sorted((Path("ts1m_ma_48_clean/train") / c).glob("*"))[:1]
    if not f: return ""
    shutil.copy2(f[0], out / "classes" / (c + f[0].suffix)); return "classes/" + c + f[0].suffix
def nice(c): return "Other sign (not in the 70 classes)" if c == "9999_other_sign" else c[5:].replace("_", " ")
items = []
with open("adjudication_key.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["item", "set", "option_A", "option_B", "A_is", "original_file"])
    for k, i in enumerate(disp, 1):
        name = f"J{k:02d}.png"; shutil.copy2(key[i]["original_file"], out / "crops" / name)
        c1, c2 = norm(key[i]["true_class"]), (norm(lab[i]) if lab[i] != "UNSURE" else None)
        opts = [(c1, "annotator1")] + ([(c2, "annotator2")] if c2 else [])
        random.shuffle(opts)
        A = opts[0]; B = opts[1] if len(opts) > 1 else ("", "")
        w.writerow([name, key[i]["set"], A[0], B[0], A[1], key[i]["original_file"]])
        items.append({"img": "crops/" + name, "name": name, "A": nice(A[0]), "Aimg": ex(A[0]), "B": nice(B[0]) if B[0] else "", "Bimg": ex(B[0]) if B[0] else ""})
html = """<!doctype html><html><head><meta charset="utf-8"><title>Adjudication</title><style>
body{font-family:Arial;margin:20px;background:#f4f4f4}.it{background:#fff;border:1px solid #ccc;border-radius:8px;padding:12px;margin-bottom:14px;display:flex;gap:18px;align-items:center}
.crop{width:200px;height:200px;object-fit:contain;background:#000}.opt{border:2px solid #ddd;border-radius:6px;padding:8px;cursor:pointer;text-align:center;width:170px}
.opt img{width:64px;height:64px;display:block;margin:0 auto 4px}.sel{border-color:#d62728;background:#ffecec}button{font-size:15px;padding:10px 16px}</style></head><body>
<h2>Which label is correct?</h2><p>For each picture, click the sign it shows. If neither option is right, click "Neither"; if the picture is too unclear, click "Cannot tell". Then click the button at the bottom.</p>
<div id="L"></div><button onclick="dl()" style="background:#2b6cb0;color:#fff">Download my choices</button>
<script>const IT=__ITEMS__;let ch={};
function draw(){const L=document.getElementById("L");L.innerHTML="";IT.forEach(t=>{const d=document.createElement("div");d.className="it";
let h='<b>'+t.name+'</b><img class="crop" src="'+t.img+'">';
const o=[["A",t.A,t.Aimg]];if(t.B)o.push(["B",t.B,t.Bimg]);o.push(["NEITHER","Neither (another sign)",""]);o.push(["UNSURE","Cannot tell",""]);
o.forEach(x=>{h+='<div class="opt'+(ch[t.name]===x[0]?' sel':'')+'" onclick="ch[\\''+t.name+'\\']=\\''+x[0]+'\\';draw()">'+(x[2]?'<img src="'+x[2]+'">':'')+x[1]+'</div>'});
d.innerHTML=h;L.appendChild(d)})}
function dl(){const t="item,choice\\n"+IT.map(x=>x.name+","+(ch[x.name]||"")).join("\\n");const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([t],{type:"text/csv"}));a.download="adjudication_choices.csv";a.click()}
draw();</script></body></html>"""
(out / "index.html").write_text(html.replace("__ITEMS__", json.dumps(items)), encoding="utf-8")
(out / "README.txt").write_text("Open index.html, choose for each picture, click 'Download my choices' and send back adjudication_choices.csv.\nPlease decide from the picture only.\n", encoding="utf-8")
print(f"adjudication_pack/: {len(items)} disputed crops; key in adjudication_key.csv (keep it private)")
