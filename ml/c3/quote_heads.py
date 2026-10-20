"""Quote heads: a Deal Kit labor proposal from a deal's pre-quote atoms and its OrbitBrief (c3_brief).

Runs beside the C3 super heads (Mac today, the T4 host in prod). Needs only the bundle file
(SOWSMITH_C3_QUOTE_HEADS) and an Ollama endpoint (OLLAMA_URL, default localhost) with the bundle's
embedding model and, for task wording, its wording model. The heads choose every number; the wording
model only words the task line and may use no number the heads did not check.

Chain, the way a quoter thinks:
  who is buying   -> account / reseller-seller profile: rate line, price, usual crew
  what the job is -> deal name + kept work lines matched to described job types
  how long        -> typical hours per tech for the job type; scaled by a checked unit count where counts drive the work
  context         -> effort or crew the documents state outright; height, after-hours, multi-site flags
"""
from __future__ import annotations
import json,math,os,re,urllib.request
from typing import Any

NUM={'a':1,'an':1,'single':1,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'twelve':12,'twenty':20}
UNITS={'tv':r'tvs?|televisions?|displays?|monitors?|screens?','camera':r'cameras?|cams?','ap':r'aps?|access points?|waps?',
 'room':r'(conference |meeting |board ?)?rooms?','drop':r'drops?|cable runs?','battery':r'batter(y|ies)|cartridges?|battery packs?','ups':r'ups(es)?|smart-ups|ups units?',
 'server':r'servers?|blades?','rack':r'racks?|cabinets?','switch':r'switch(es)?','phone':r'phones?|handsets?','workstation':r'workstations?|desks?|pcs?|laptops?|computers?',
 'site':r'sites?|locations?|stores?|hotels?|offices?','user':r'users?','device':r'devices?'}
PAT={u:re.compile(r"\(?\b(\d{1,4}|"+"|".join(NUM)+r")\)?\s*(\(\d+\)\s*)?(new\s+|existing\s+|additional\s+)?(?:[\w\"”'-]+\s+){0,3}?(?:"+p+r")\b",re.I) for u,p in UNITS.items()}
SIZE=re.compile(r'\d\s*("|”|-inch|inch|in\.|°|%|ft\b|\'|v\b|u\b|-?port|w\b|watt|mm\b|gb\b|tb\b|kw|kva|amp|x\d)',re.I)
QTYX=re.compile(r"(?:qty|quantity|qty\.)\s*[:=]?\s*(\d{1,4})\b|col_\d+:\s*(\d{1,4})\s*\|",re.I)
PART=re.compile(r'[A-Z]{2,}\d{2,}|#\d+|\d+[A-Z]{2,}|model',re.I)
EFF=re.compile(r"(?:approximately|approx\.?|about|around|roughly|duration:?)\s*(\d{1,2})(?:\s*(?:-|to)\s*(\d{1,2}))?\s*hours?|(\d{1,2})(?:\s*(?:-|to)\s*(\d{1,2}))?\s*hours?\s*(?:per|each|a)\s*(?:property|site|location|shift|day|tech|technician)|(?:a|one)\s+(?:guy|tech|technician)\s+for\s+(\d{1,2})(?:\s*-\s*(\d{1,2}))?\s*hours?",re.I)
TECHS=re.compile(r"\b(\d|one|two|three|four)\s*(?:\(\d\)\s*)?(?:onsite\s+|field\s+|dedicated\s+)?(?:techs?|technicians?|engineers?|installers?)\b",re.I)
HEIGHT=re.compile(r'\b(lift|scissor|boom|roof|mast|\d{2}\s*(?:-\s*\d{2}\s*)?(?:ft|feet|foot|\'))\b',re.I)
AFTER=re.compile(r'\bafter[- ]hours\b|\bovernight\b|\bweekends?\b|\bnights?\b',re.I)
NOISE=re.compile(r'net\s*\d+|warranty|licen|within \d+ (?:hours|days)|notice|24-48|24\s*/\s*7|quotes? (?:delivered|in)|business days|window',re.I)
HB=(2,4,6,8,10,12,16,24,32,40,48,64,80,120,200,400,600)
SKIP_TYPES={'small_talk','stakeholder','deal_metadata'}

def _snap(h): return min(HB,key=lambda x:abs(math.log(x)-math.log(max(h,1))))

class QuoteHeads:
    def __init__(self,bundle_path:str,ollama:str|None=None):
        self.b=json.load(open(bundle_path)); self.ollama=(ollama or os.environ.get('OLLAMA_URL') or 'http://localhost:11434').rstrip('/')
        self.fams=list(self.b['descriptions']); self.D=[self._norm(v) for v in self.b['description_embeddings']]
        self.version=self.b['version']
    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _norm(v):
        n=math.sqrt(sum(x*x for x in v)) or 1.0; return [x/n for x in v]
    def _post(self,path,body,timeout=120):
        req=urllib.request.Request(self.ollama+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(req,timeout=timeout))
    def _embed(self,texts):
        out=self._post('/api/embed',{'model':self.b['embed_model'],'input':['Instruct: Identify the type of onsite IT field-services job.\nQuery: '+t for t in texts]})
        return [self._norm(v) for v in out['embeddings']]
    # -- reading the deal ---------------------------------------------------
    def _work(self,p):
        brief=p.get('c3_brief') or {}; secs=brief.get('sections') or {}
        kept={x.get('id') for x in (secs.get('scope_of_work') or [])}
        cited={c for j in (brief.get('job_details') or []) for c in (j.get('cites') or [])}
        work=[]
        for a in p.get('atoms') or []:
            t=(a.get('text') or '').strip(); ty=a.get('type') or a.get('atom_type')
            if not t or ty in SKIP_TYPES: continue
            if a.get('id') in kept or a.get('id') in cited or ty=='deal_summary': work.append(a)
        return work,cited
    def _quantity(self,p,work,cited):
        name=(p.get('deal_name') or '').lower(); groups={}
        wid={a.get('id') for a in work}
        for a in p.get('atoms') or []:
            if (a.get('type') or a.get('atom_type')) in SKIP_TYPES: continue
            t=a.get('text') or ''
            found=[]
            for u,rx in PAT.items():
                for m in rx.finditer(t):
                    if SIZE.match(t[m.start(1):m.end(1)+7]): continue
                    n=m.group(1).lower()
                    if n in ('a','an') and u in ('site','user','device','room'): continue
                    n=NUM.get(n,n)
                    try: n=int(n)
                    except ValueError: continue
                    if 0<n<=300: found.append((u,n,t[max(0,m.start()-40):m.end()+40]))
            for m in QTYX.finditer(t):
                n=int(m.group(1) or m.group(2))
                if not 0<n<=300: continue
                for u in UNITS:
                    if u not in ('site','user') and re.search(r'\b(?:'+UNITS[u]+r')\b',t,re.I): found.append((u,n,t[max(0,m.start()-60):m.end()+10])); break
            for u,n,txt in found:
                g=groups.setdefault((u,n),dict(unit=u,count=n,n=0,labor=0,cited=0,part=0,text=txt,atom_id=a.get('id')))
                g['n']+=1; g['labor']|=a.get('id') in wid; g['cited']|=a.get('id') in cited; g['part']|=bool(PART.search(txt))
        w=self.b['quantity_weights']; best=None; scores=[]
        for g in groups.values():
            inname=bool(re.search(r'\b('+UNITS[g['unit']]+r')\b',name,re.I))
            f=[1.0,math.log1p(g['count']),math.log1p(g['n']),float(g['labor']),float(g['cited']),0.0,float(g['part']),float(g['count']>=100),float(inname),float(g['count']==1)]
            s=sum(a*b for a,b in zip(w,f)); scores.append((s,g))
        if not scores: return None
        mx=max(s for s,_ in scores); z=sum(math.exp(s-mx) for s,_ in scores)
        s,g=max(scores,key=lambda x:x[0]); g=dict(g); g['p']=math.exp(s-mx)/z
        return g
    def _stated(self,work):
        hs=[];ts=[]
        for a in work:
            t=a.get('text') or ''
            if NOISE.search(t): continue
            for m in EFF.finditer(t):
                g=[int(x) for x in m.groups() if x]; hs.append(sum(g)/len(g))
            for m in TECHS.finditer(t):
                v=m.group(1).lower(); ts.append(NUM.get(v) or int(v))
        h=max(hs) if hs else None
        return (max(ts) if ts else None),(h if h and h<=16 else None)
    def _profile(self,p):
        P=self.b['profiles']; acc=(p.get('account') or '').strip()
        emails={m.group(0).lower() for a in p.get('atoms') or [] for m in re.finditer(r'[\w.\-]+@[\w\-]+\.(?:com|net|org|io)',a.get('text') or '') if 'purtera' not in m.group(0).lower()}
        seller=next((P[f'seller:{e}'] for e in sorted(emails) if f'seller:{e}' in P),None)
        account=None
        for k,v in P.items():
            if v['kind']=='account' and acc and (v['key'].lower()==acc.lower() or v['key'].lower() in (p.get('deal_name') or '').lower()): account=v; break
        return seller,account
    # -- the proposal -------------------------------------------------------
    def quote(self,p:dict[str,Any],word:bool=True)->dict[str,Any]:
        work,cited=self._work(p)
        name=re.sub(r'^\d{6}\s*-\s*','',p.get('deal_name') or '')
        scope=name+'. '+' '.join((a.get('text') or '')[:250] for a in work[:10])
        ej,en=self._embed([scope,name or scope])
        sims=[0.7*sum(x*y for x,y in zip(ej,d))+0.3*sum(x*y for x,y in zip(en,d)) for d in self.D]
        fam=self.fams[max(range(len(sims)),key=lambda i:sims[i])]; F=self.b['families'][fam]
        seller,account=self._profile(p); prof=seller or (account if account and account['kits']>=2 else None)
        st_t,st_h=self._stated(work); q=self._quantity(p,work,cited)
        # techs
        if st_t: techs=min(st_t,4); why_t='stated in the documents'
        elif prof and prof.get('usual_techs') and (prof.get('techs_share') or 0)>=0.75 and prof['kits']>=2: techs=prof['usual_techs']; why_t=f"{prof['key']} usually gets {prof['usual_techs']}"
        else:
            ut={int(k):v for k,v in F['usual_techs'].items()}; techs=max(ut,key=ut.get) if ut else 1; why_t=f'usual crew for {fam} jobs'
        # hours per tech
        n=q['count'] if q and q['unit'] not in ('site','user') and q['p']>=0.5 else None
        if st_h: hpt=st_h; why_h='stated in the documents'
        elif n and F.get('per_unit_crew_hours') and fam in self.b['count_scaled_families'] and n<=(300 if fam in ('ap','camera','power') else 60):
            hpt=max(min(F['median_hours_per_tech'],8),n*F['per_unit_crew_hours']/techs); why_h=f'{n} {q["unit"]}s at the usual time each for {fam} jobs'
        else: hpt=F['median_hours_per_tech']; why_h=f'typical for {fam} jobs'
        hpt=_snap(hpt)
        # rate line and price
        rp=seller or (account if account and account['rate_line_share']>=0.6 else None)
        rate_line=rp['rate_line'] if rp else F['usual_rate_line']; price=rp['price'] if rp else F['usual_price']
        why_r=f"{rp['key']} profile ({rp['kits']} kits)" if rp else f'usual for {fam} jobs'
        alltext=' '.join(a.get('text') or '' for a in work)
        flags=[]
        if AFTER.search(alltext): flags.append('after-hours work mentioned')
        if HEIGHT.search(alltext): flags.append('height / lift access mentioned')
        if (p.get('site_count') or 0)>=2: flags.append(f"{p['site_count']} places in the documents: check the site list")
        if fam=='program': flags.append('ongoing or multi-site engagement: confirm the duration and site count')
        checked=(q['unit'],q['count']) if q and q['p']>=0.5 and q['unit'] not in ('site','user','device') and re.search(r'\b('+UNITS[q['unit']]+r')\b',scope,re.I) else None
        task=self._word(name,work,p.get('c3_brief') or {},checked,p.get('account')) if word else None
        verdict=f"hours={hpt*techs:g};per=job;qty=1;role={rate_line};techs={techs};hpt={hpt:g}"
        return dict(version=self.version,job_type=fam,job_type_p=round(max(sims),3),
            lines=[dict(site='job site',task=task or {'av':'Install AV equipment','camera':'Install cameras','ap':'Install access points','power':'Replace UPS / batteries','rack':'Rack and stack equipment','cabling':'Run network cabling','device':'Install and configure devices','survey':'Site survey and report','program':'Onsite support (staff augmentation)'}[fam],
                techs=techs,hours_per_tech=hpt,crew_hours=hpt*techs,rate_line=rate_line,price=price,why_techs=why_t,why_hours=why_h,why_rate=why_r,proposed_hours=verdict)],
            quantity=({'unit':q['unit'],'count':q['count'],'p':round(q['p'],2),'text':q['text'],'atom_id':q['atom_id']} if q else None),flags=flags)
    def _word(self,name,work,brief,checked,account=None):
        jd=[(j.get('text') or '').strip() for j in (brief.get('job_details') or [])][:4]
        facts=f"Deal name: {name}\nJob details: {' | '.join(j[:220] for j in jd) or ' | '.join((a.get('text') or '')[:200] for a in work[:4]) or '(none)'}\n"
        if checked: facts+=f"Checked count: {checked[1]} {checked[0]}\n"
        prompt=("Write ONE Deal Kit labor task line for this field-services job, at most 9 words, starting with a verb. Use only the facts given and describe the work "
                "the customer is asking us to do (not drawings, not other projects). The deal name is the most reliable statement of what the job is. The ONLY number allowed "
                "is the checked count (if one is given, use it exactly); no other digits or number words. No customer, street or city names, no dates, weekdays or months, "
                "no prices, no model or part numbers, no punctuation at the end.\n\n"+facts+"\nTask line:")
        try: out=self._post('/api/generate',{'model':self.b['wording_model'],'prompt':prompt,'stream':False,'think':False,'options':{'temperature':0}})['response'].strip().split('\n')[0]
        except Exception: return None
        out=re.sub(r'^[\-\*"\s]+|["\.\s]+$','',out)
        allowed={str(checked[1])} if checked else set()
        WORDS=r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|single)\b'
        DATES=r'\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|\b(mon|tues|wednes|thurs|fri|satur|sun)day\b'
        def bad(t): return any(n not in allowed for n in re.findall(r'\d[\d,]*',t)) or bool(re.search(DATES,t,re.I)) or (bool(re.search(WORDS,t,re.I)) and not checked) or bool(PART.search(t)) or ' - ' in t
        if bad(out):
            fix=("Rewrite this task line with no digits, sizes, model or part numbers, street names, dates or counts"+(f" except the count {checked[1]}" if checked else "")+
                 ". Under 9 words, start with a verb, plain words a project manager would use, nothing else:\n"+out)
            try: out=re.sub(r'^[\-\*"\s]+|["\.\s]+$','',self._post('/api/generate',{'model':self.b['wording_model'],'prompt':fix,'stream':False,'think':False,'options':{'temperature':0}})['response'].strip().split('\n')[0])
            except Exception: pass
        out=' '.join(w for w in out.split() if not (re.search(r'\d',w) and re.sub(r'\D','',w) not in allowed))
        # never echo the deal name or the account; a plain line beats a wrong one
        name_words={w.lower() for w in re.findall(r'[A-Za-z]{4,}',(name or '')+' '+(account or '')) if w.lower() not in {'install','installation','camera','cameras','office','support','request','smart','hands','phone','phones','display','displays','battery','cabling','network','rack','switch','survey','room','rooms','site','sites','multi','location','deployment','migration','relocation','replacement','upgrade','refresh'}}
        leaked=[w for w in re.findall(r'[A-Za-z]{4,}',out) if w.lower() in name_words]
        if bad(out) or len(leaked)>=2 or ' - ' in out: return None
        if len(out)>90: out=out[:90].rsplit(' ',1)[0]
        return out or None
