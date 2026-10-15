"""Render c3_record/heads.json as the labeling guide page: python -m ml.c3.heads_page c3_record/heads.json docs/super_heads_registry.html"""
import json, sys, html
reg = json.load(open(sys.argv[1]))["heads"]; out = sys.argv[2]
data = json.dumps(reg)
page = r"""<title>Super Heads Registry</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600&display=swap">
<style>
/* Layout: one column ledger; summary strip, ranked labeling queue, then a filterable table per layer */
:root{--bg:#f6f7f5;--panel:#ffffff;--fg:#1d2420;--muted:#5d6b63;--line:#dfe4e0;--accent:#2f6f5e;
--live:#2f6f5e;--ready:#3b7d3a;--weak:#a86b12;--warn:#b5521f;--none:#8a948f;
--f-display:'Fraunces',Georgia,serif;--f-body:'IBM Plex Sans',system-ui,sans-serif;--f-mono:'IBM Plex Mono',ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121614;--panel:#1a201d;--fg:#e4ebe7;--muted:#97a69e;--line:#2b342f;--accent:#7cc4ad;--live:#7cc4ad;--ready:#8fcf7c;--weak:#e0a84a;--warn:#ec8a5a;--none:#7d8781;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121614;--panel:#1a201d;--fg:#e4ebe7;--muted:#97a69e;--line:#2b342f;--accent:#7cc4ad;--live:#7cc4ad;--ready:#8fcf7c;--weak:#e0a84a;--warn:#ec8a5a;--none:#7d8781;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.5 var(--f-body);padding:24px 16px 48px}
.wrap{max-width:1100px;margin:0 auto;display:grid;gap:28px}
h1{font:600 2rem/1.15 var(--f-display);margin:0;text-wrap:balance}
h2{font:600 1.15rem/1.3 var(--f-body);margin:0 0 10px}
p{margin:0;max-width:68ch;color:var(--muted)}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.cell b{font:500 1.4rem var(--f-mono);display:block;font-variant-numeric:tabular-nums}
.cell span{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.pill{display:inline-block;font:500 .72rem var(--f-mono);padding:1px 7px;border-radius:99px;border:1px solid currentColor;white-space:nowrap}
.s-live{color:var(--live)}.s-ready{color:var(--ready)}.s-weak,.s-unproven{color:var(--weak)}.s-one_sided,.s-thin{color:var(--warn)}.s-no_labels{color:var(--none)}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.tabs button,.filters select{font:500 .9rem var(--f-body);background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:6px 12px;cursor:pointer}
.tabs button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent)}
button:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.tbl{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;min-width:860px;font-size:.88rem}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:.74rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:500}
td.k{font-family:var(--f-mono);white-space:nowrap}
td.n{font-family:var(--f-mono);font-variant-numeric:tabular-nums;white-space:nowrap}
ul.rep{margin:0;padding-left:16px;color:var(--muted)}
ol.queue{margin:0;padding-left:22px;display:grid;gap:8px}
ol.queue li{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:9px 12px}
.q-head{font-family:var(--f-mono);font-weight:500}
.sub{color:var(--muted);font-size:.86rem}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.85rem;color:var(--muted)}
</style>
<div class="wrap">
<header style="display:grid;gap:8px">
<h1>Super Heads Registry</h1>
<p>Every head the C3 base computes, sorted into the <b>base</b> layer (what a line is, the same for any company) and the <b>Purtera</b> layer (what Purtera does with it). Status is computed from the unseen holdout and the label files, never set by hand. A head that is <i>live</i> or <i>ready</i> decides its stage; anything else runs on its fallback until labeling promotes it.</p>
</header>
<section><h2>Where the heads stand</h2><div class="strip" id="strip"></div></section>
<section><h2>Label next</h2><p style="margin-bottom:10px">Ranked by how much parser and OrbitBrief work a head would take over once its labels are fixed.</p><ol class="queue" id="queue"></ol></section>
<section>
<h2>All heads</h2>
<div class="legend" style="margin-bottom:10px">
<span><span class="pill s-live">live</span> wired in, deciding</span><span><span class="pill s-ready">ready</span> beats guessing by 10+ pts</span><span><span class="pill s-weak">weak</span> near guess level</span><span><span class="pill s-unproven">unproven</span> not scored</span><span><span class="pill s-one_sided">one_sided</span> labels all one answer</span><span><span class="pill s-thin">thin</span> under 100 labels or 5 deals</span><span><span class="pill s-no_labels">no_labels</span></span></div>
<div class="tabs" role="group" aria-label="Layer">
<button id="t-base" aria-pressed="true" data-l="base">Base layer</button>
<button id="t-purtera" aria-pressed="false" data-l="purtera">Purtera layer</button>
<label class="filters" style="margin-left:auto">Status <select id="f-status"><option value="">all</option><option>live</option><option>ready</option><option>weak</option><option>unproven</option><option>one_sided</option><option>thin</option><option>no_labels</option></select></label>
<label class="filters"><input type="checkbox" id="f-rep"> replaces a stage</label>
</div>
<div class="tbl"><table><thead><tr><th>Head</th><th>Status</th><th>Replaces</th><th>Fallback</th><th>Holdout vs guess</th><th>Labels / deals</th><th>Labeling task</th></tr></thead><tbody id="rows"></tbody></table></div>
</section>
</div>
<script>
const R=__DATA__;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pill=s=>`<span class="pill s-${s}">${s.replace('_',' ')}</span>`;
const st={};R.forEach(h=>{const k=h.layer+'|'+h.status;st[k]=(st[k]||0)+1});
const cells=[['base','live'],['base','ready'],['purtera','ready'],['base','weak'],['base','one_sided'],['purtera','one_sided']].map(([l,s])=>`<div class="cell"><b>${st[l+'|'+s]||0}</b><span>${l} · ${s.replace('_',' ')}</span></div>`);
const nb=R.filter(h=>h.layer==='base').length,np=R.length-nb;
document.getElementById('strip').innerHTML=`<div class="cell"><b>${nb}</b><span>base heads</span></div><div class="cell"><b>${np}</b><span>purtera heads</span></div>`+cells.join('');
const rank={one_sided:0,weak:1,unproven:2,thin:3,no_labels:4};
const q=R.filter(h=>h.replaces.length&&h.status in rank).sort((a,b)=>b.replaces.length-a.replaces.length||rank[a.status]-rank[b.status]).slice(0,14);
document.getElementById('queue').innerHTML=q.map(h=>`<li><span class="q-head">${esc(h.key)}</span> ${pill(h.status)} <span class="sub">${h.layer}</span><div>${esc(h.labeling_task)}</div><div class="sub">Would replace: ${h.replaces.map(esc).join('; ')}</div></li>`).join('');
let layer='base';try{layer=localStorage.getItem('reg-layer')||'base'}catch(e){}
function draw(){const fs=document.getElementById('f-status').value,fr=document.getElementById('f-rep').checked;
document.querySelectorAll('.tabs button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.l===layer));
document.getElementById('rows').innerHTML=R.filter(h=>h.layer===layer&&(!fs||h.status===fs)&&(!fr||h.replaces.length)).map(h=>{
const ho=h.holdout?`${(h.holdout.exact*100).toFixed(0)}%${h.holdout.guess!=null?' vs '+(h.holdout.guess*100).toFixed(0)+'%':''} <span class="sub">n=${h.holdout.n}</span>`:'<span class="sub">not scored</span>';
return `<tr><td class="k">${esc(h.key)}</td><td>${pill(h.status)}</td><td>${h.replaces.length?'<ul class="rep">'+h.replaces.map(r=>'<li>'+esc(r)+'</li>').join('')+'</ul>':'<span class="sub">none yet</span>'}</td><td>${esc(h.fallback)||'<span class="sub">none</span>'}</td><td class="n">${ho}</td><td class="n">${h.labels.toLocaleString()} / ${h.label_deals}</td><td>${esc(h.labeling_task)}</td></tr>`}).join('')}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{layer=b.dataset.l;try{localStorage.setItem('reg-layer',layer)}catch(e){};draw()});
document.getElementById('f-status').onchange=draw;document.getElementById('f-rep').onchange=draw;draw();
</script>"""
open(out,"w").write(page.replace("__DATA__",data))
