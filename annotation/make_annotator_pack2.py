"""REVISION: pack 2 for the SECOND annotator (Omar El Azzouzi): every remaining test crop.

Pack 1 (make_annotator_pack.py) held a random sample of 500 crops (200 final + 200 development + 100 Tangier).
Pack 2 holds ALL the other crops of the five test sets, including the signs outside our 70 classes:
development (morocco_youtube), final test (morocco_final), Tangier dashcam (morocco_dashcam) and the two rain
drives (morocco_rain_rabat, morocco_rain_tanger). Crops already in pack 1 (annotator_key.csv) are left out.
Same blind tool as pack 1: random order, neutral names B0001.png ..., no labels and no model output shown.

Usage (project folder):  python make_annotator_pack2.py            (all remaining crops, about 1,170)
                         python make_annotator_pack2.py --rain-only (only the two rain drives, 247 crops)
Gives:  annotator_pack2/            -> zip this folder and send it to the second annotator
        annotator_key_part2.csv     -> KEEP THIS (true labels); do NOT send it
"""
import csv, json, random, shutil
from pathlib import Path

SETS = {"morocco_youtube": "dev", "morocco_final": "final", "morocco_dashcam": "tangier",
        "morocco_rain_rabat": "rain_rabat", "morocco_rain_tanger": "rain_tanger"}
import sys
if "--rain-only" in sys.argv:
    SETS = {k: v for k, v in SETS.items() if v.startswith("rain")}
OTHER = "9999_other_sign"
classes = sorted(d.name for d in Path("ts1m_ma_48_clean/train").iterdir() if d.is_dir())
done = {r["original_file"] for r in csv.DictReader(open("annotator_key.csv", encoding="utf-8"))}
out = Path("annotator_pack2"); shutil.rmtree(out, ignore_errors=True)
(out / "crops").mkdir(parents=True); (out / "classes").mkdir()
random.seed(2027)
pool = []
for folder in SETS:
    pool += [(folder, d.name, f) for d in sorted((Path(folder) / "test").iterdir()) if d.is_dir()
             for f in sorted(d.glob("*.png")) if (d.name in classes or d.name == OTHER) and f.as_posix() not in done]
random.shuffle(pool)
with open("annotator_key_part2.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["item", "set", "true_class", "original_file"])
    for i, (folder, cls, f) in enumerate(pool, 1):
        name = f"B{i:04d}.png"; shutil.copy2(f, out / "crops" / name)
        w.writerow([name, SETS[folder], cls, f.as_posix()])
cls_list = []
for c in classes:
    ex = sorted((Path("ts1m_ma_48_clean/train") / c).glob("*"))[:1]
    if ex:
        shutil.copy2(ex[0], out / "classes" / (c + ex[0].suffix)); img = "classes/" + c + ex[0].suffix
    else:
        img = ""
    cls_list.append({"id": c, "name": c[5:].replace("_", " "), "img": img})
items = [f"B{i:04d}.png" for i in range(1, len(pool) + 1)]
html = """<!doctype html><html><head><meta charset="utf-8"><title>Sign labelling - part 2</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#f4f4f4;color:#222}
#top{position:sticky;top:0;background:#fff;padding:10px 16px;border-bottom:1px solid #ccc;display:flex;gap:20px;align-items:center;z-index:2}
#crop{width:260px;height:260px;object-fit:contain;image-rendering:auto;background:#000;border:3px solid #333}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;padding:12px 16px}
.c{background:#fff;border:2px solid #ddd;border-radius:6px;padding:6px;text-align:center;cursor:pointer;font-size:12px}
.c:hover{border-color:#2b6cb0}.c img{width:64px;height:64px;image-rendering:pixelated;display:block;margin:0 auto 4px}
.sel{border-color:#d62728;background:#ffecec}.big{font-size:15px;padding:12px;font-weight:bold}
button{font-size:14px;padding:8px 14px;margin-right:6px;cursor:pointer}
#filter{font-size:14px;padding:6px;width:220px}
</style></head><body>
<div id="top"><img id="crop"><div>
<div style="font-size:18px"><b id="pos"></b></div>
<p style="max-width:520px;font-size:13px">Click the sign that matches the picture. Use <b>Other sign</b> if it is a real traffic sign that is not in the list,
and <b>Cannot tell</b> if the picture is too unclear. Your answers are saved in this browser automatically.</p>
<button onclick="go(-1)">&larr; Previous</button><button onclick="go(1)">Next &rarr;</button>
<button onclick="dl()" style="background:#2b6cb0;color:#fff">Download my labels</button>
<input id="filter" placeholder="search a sign name..." oninput="draw()"></div></div>
<div id="grid"></div>
<script>
const CL=__CLASSES__, IT=__ITEMS__, KEY="sign_labels_part2_v1";
let lab={}; try{lab=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0; while(i<IT.length-1 && lab[IT[i]]) i++;
function save(){try{localStorage.setItem(KEY,JSON.stringify(lab))}catch(e){}}
function draw(){
 document.getElementById("crop").src="crops/"+IT[i];
 const done=IT.filter(x=>lab[x]).length;
 document.getElementById("pos").textContent="Image "+(i+1)+" of "+IT.length+"  ("+done+" labelled)";
 const f=document.getElementById("filter").value.toLowerCase(), g=document.getElementById("grid"); g.innerHTML="";
 const extra=[{id:"OTHER",name:"Other sign (not in the list)",img:""},{id:"UNSURE",name:"Cannot tell",img:""}];
 for(const c of extra.concat(CL)){ if(f && !c.name.toLowerCase().includes(f) && c.img) continue;
  const d=document.createElement("div"); d.className="c"+(c.img?"":" big")+(lab[IT[i]]===c.id?" sel":"");
  d.innerHTML=(c.img?'<img src="'+c.img+'">':"")+c.name; d.onclick=()=>{lab[IT[i]]=c.id;save();go(1)}; g.appendChild(d);}
}
function go(s){i=Math.max(0,Math.min(IT.length-1,i+s));document.getElementById("filter").value="";draw();window.scrollTo(0,0)}
function dl(){let t="item,label\\n"+IT.map(x=>x+","+(lab[x]||"")).join("\\n");
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([t],{type:"text/csv"}));a.download="labels_annotator2_part2.csv";a.click()}
document.addEventListener("keydown",e=>{if(e.target.id==="filter")return;if(e.key==="ArrowRight")go(1);if(e.key==="ArrowLeft")go(-1)});
draw();
</script></body></html>"""
(out / "index.html").write_text(html.replace("__CLASSES__", json.dumps(cls_list)).replace("__ITEMS__", json.dumps(items)), encoding="utf-8")
(out / "README.txt").write_text(
    "1. Unzip this folder.\n2. Double-click index.html (opens in your web browser).\n"
    "3. For each picture, click the matching sign (or 'Other sign' / 'Cannot tell').\n"
    "   You can stop and come back later: answers are kept in the same browser.\n"
    "4. At the end, click 'Download my labels' and send the file labels_annotator2_part2.csv back.\n"
    "   (You can also download it part-way through as a backup; the last download is the one that counts.)\n"
    "Please work alone and do not look at any other labels.\n", encoding="utf-8")
print(f"annotator_pack2/: {len(pool)} crops, {len(cls_list)} classes; key in annotator_key_part2.csv (keep it private); left out {len(done)} crops of pack 1")
