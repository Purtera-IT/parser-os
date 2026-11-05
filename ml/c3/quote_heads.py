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
#: A head that answered nothing for a line, or answered "there is no such fact here".
ABSENT=('_absent','','none',None)
#: The window labels that mean the work itself happens outside business hours. ``blackout`` is a
#: window the work must AVOID, not one it happens in, so it never raises the after-hours flag.
OFF_HOURS=('after_hours','overnight','weekend','holiday')
LEVEL=re.compile(r'\b(?:l|level|tier)\s*-?\s*([123])\b',re.I)

def _said(a,head):
    """What the super heads read on one line: ``(answer, p, value)``, or None when they did not
    answer.

    ``answer`` is a label, or a list of labels for a multi-label head (read:work_window). The
    confidence arrives two ways and both are real: a scalar for a single-answer head, and a
    ``{label: p}`` map for a multi-label one -- reading that map with ``float()`` makes every
    multi-label p zero, which is what the first cut of this did ("p 0.00" on a window the head gave
    0.58). ``p`` here is the strongest asserted label's. ``value`` is the number a NUMBER head sent.
    """
    v=(a.get('reads') or {}).get(head) if isinstance(a.get('reads'),dict) else None
    if not isinstance(v,(list,tuple)) or not v or v[0] is None: return None
    conf=v[1] if len(v)>1 else 0.0
    labs=_labels(v[0])
    if isinstance(conf,dict):
        ps=[float(conf.get(l) or 0.0) for l in labs]
        p=max(ps) if ps else 0.0
    else:
        try: p=float(conf)
        except (TypeError,ValueError): p=0.0
    return v[0],p,(v[2] if len(v)>2 else None)


def _p_of(a,head,label):
    """How sure the head is of one label on this line."""
    v=(a.get('reads') or {}).get(head) if isinstance(a.get('reads'),dict) else None
    if not isinstance(v,(list,tuple)) or len(v)<2: return 0.0
    if isinstance(v[1],dict): return float(v[1].get(label) or 0.0)
    try: return float(v[1])
    except (TypeError,ValueError): return 0.0

def _labels(ans):
    """A head's answer as the labels it actually asserts, minus the absent ones."""
    xs=ans if isinstance(ans,(list,tuple)) else [ans]
    return [x for x in xs if x not in ABSENT]

def _present(a,head):
    """This line holds the fact `head` is about, and how sure the head is."""
    got=_said(a,head)
    return (True,got[1]) if got and _labels(got[0]) else (False,0.0)

def _denies(a,head):
    """The head read this line and said the fact is NOT here. A regex match on such a line is a
    false positive the head already ruled out -- "within 24 hours", "net 30 days", a warranty
    window. This is the head vetoing the rule, which is the point of having the head."""
    got=_said(a,head)
    return bool(got) and not _labels(got[0])

def _marked(atoms,head,p_min=0.0):
    """Every line the head says holds this fact, surest first. The heads decide which lines speak;
    the extraction below only reads a number off a line one of them picked."""
    hits=[(p,a) for a in atoms if (g:=_present(a,head))[0] and (p:=g[1])>=p_min]
    hits.sort(key=lambda t:-t[0])
    return [a for _p,a in hits]


def _snap(h): return min(HB,key=lambda x:abs(math.log(x)-math.log(max(h,1))))

class QuoteHeads:
    def __init__(self,bundle_path:str,ollama:str|None=None):
        self.b=json.load(open(bundle_path)); self.ollama=(ollama or os.environ.get('OLLAMA_URL') or 'http://localhost:11434').rstrip('/')
        self.fams=list(self.b['descriptions']); self.D=[self._norm(v) for v in self.b['description_embeddings']]
        self.version=self.b['version']
        # Wording model: SOWSMITH_C3_QUOTE_WORDING = off | <ollama model>; default the bundle's (off on a T4).
        w=os.environ.get('SOWSMITH_C3_QUOTE_WORDING','').strip()
        self.wording=None if w.lower()=='off' else (w or self.b.get('wording_model'))
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
    @staticmethod
    def _read_number(a,head,rx,cap=None):
        """One line's number for one head. A NUMBER head sends its own value and that value stands.
        Otherwise the head says only whether the fact is HERE (present/_absent) and `rx` reads the
        number off the line it marked -- heads decide, the parser extracts."""
        got=_said(a,head)
        if got and got[2] is not None:
            try:
                v=float(got[2])
                return v if (cap is None or 0<v<=cap) else None
            except (TypeError,ValueError): pass
        t=a.get('text') or ''
        if NOISE.search(t): return None
        vals=[]
        for m in rx.finditer(t):
            g=[int(x) for x in m.groups() if x and str(x).isdigit()] or \
              [NUM[x.lower()] for x in m.groups() if x and str(x).lower() in NUM]
            if g: vals.append(sum(g)/len(g))
        vals=[v for v in vals if cap is None or 0<v<=cap]
        return max(vals) if vals else None

    def _from_heads(self,atoms,work,head,rx,cap):
        """The number this head's lines state, and where it came from.

        Three uses of the head, all of them things the regex alone got wrong:

        * **veto** -- a line the head read as ``_absent`` is skipped even when the regex matches
          it. "Response within 24 hours", "net 30" and a 48-hour warranty window stop being the
          job's effort.
        * **order** -- among the lines the head marked, the surest is read first.
        * **widen** -- when no line of the brief's work section states the fact, the head's own
          marked lines are read instead of falling straight to a family median. Deal 010452 was
          priced at a family median while the hours head had marked 1019 lines; nothing downstream
          ever looked at them.
        """
        wid={a.get('id') for a in work}
        inside=[a for a in work if not _denies(a,head)]
        for a in inside:
            if (v:=self._read_number(a,head,rx,cap)) is not None:
                said=_present(a,head)
                return v,('the work lines'+(f' ({head} agrees)' if said[0] else '')),a.get('id'),(head if said[0] else None)
        for a in _marked(atoms,head):
            if a.get('id') in wid or a.get('type') in SKIP_TYPES or a.get('left_out'): continue
            if (v:=self._read_number(a,head,rx,cap)) is not None:
                return v,f'{head} marked this line',a.get('id'),head
        return None,None,None,None

    def _window(self,atoms,work):
        """What the day-type head read, best line per label. Replaces the after-hours wording rule:
        the rule fires on the word in an EXCLUSION ("after-hours work is excluded", "no weekend
        access") and on a completion deadline that falls on a Sunday. The head answers the question
        the flag is asking -- when does the work happen -- and ``blackout`` is a window to avoid,
        which the rule had no way to tell from one to work in."""
        wid={a.get('id') for a in work}
        best={}
        for a in atoms:
            if a.get('left_out') or a.get('type') in SKIP_TYPES: continue
            got=_said(a,'read:work_window')
            if not got: continue
            for lab in _labels(got[0]):
                rank=(a.get('id') in wid,_p_of(a,'read:work_window',lab))
                if lab not in best or rank>best[lab][0]: best[lab]=(rank,a.get('id'))
        return {lab:{'p':round(r[1],3),'in_work':bool(r[0]),'line':aid} for lab,(r,aid) in best.items()}

    @staticmethod
    def _billing(atoms,work,deal):
        """The billing shape the heads read. The deal-size head answers once for the deal and that
        answer stands; otherwise the line head is counted over the work lines, by confidence."""
        d=deal.get('jdg:billing_type') if isinstance(deal,dict) else None
        if isinstance(d,(list,tuple)) and d and d[0] not in ABSENT:
            try: p=float(d[1])
            except Exception: p=0.0
            return str(d[0]),round(p,3),'jdg:billing_type (the deal)',None
        wid={a.get('id') for a in work}
        score={}
        for a in atoms:
            if a.get('id') not in wid: continue
            got=_said(a,'read:billing_type')
            if not got: continue
            for lab in _labels(got[0]):
                cur=score.get(lab) or (0.0,None)
                score[lab]=(cur[0]+got[1],cur[1] or a.get('id'))
        if not score: return None,0.0,None,None
        total=sum(v[0] for v in score.values()) or 1.0
        ranked=sorted(((v[0]/total,lab,v[1]) for lab,v in score.items()),reverse=True)
        share,lab,aid=ranked[0]
        runner=ranked[1][0] if len(ranked)>1 else 0.0
        # A plurality is not an answer. 010358's work lines split fixed 0.496 / t_and_m 0.44; calling
        # that "fixed" is how the summary came to read "Fixed fee" over a t_and_m deal. A split says
        # it is split, and the PM decides -- a head that is unsure must not sound sure.
        if share<0.55 or share-runner<0.10:
            return None,round(share,3),f'read:billing_type is split ({lab} {share:.0%} / {ranked[1][1]} {runner:.0%})' \
                if len(ranked)>1 else 'read:billing_type is unsure',aid
        return lab,round(share,3),'read:billing_type over the work lines',aid

    @staticmethod
    def _level(atoms,work):
        """The tech level a line the level head marked states outright (L1/L2/L3)."""
        wid={a.get('id') for a in work}
        for a in sorted(_marked(atoms,'read:tech_level'),key=lambda a:(a.get('id') not in wid,)):
            if a.get('left_out'): continue
            if m:=LEVEL.search(a.get('text') or ''): return int(m.group(1)),a.get('id')
        return None,None

    def _stated(self,work,atoms=None):
        """Crew and effort the documents state, read off the lines the heads marked."""
        atoms=work if atoms is None else atoms
        t_v,t_why,t_line,t_head=self._from_heads(atoms,work,'read:crew_size',TECHS,4)
        h_v,h_why,h_line,h_head=self._from_heads(atoms,work,'read:labor_hours',EFF,16)
        return (dict(v=int(t_v),why=t_why,line=t_line,head=t_head) if t_v else None), \
               (dict(v=h_v,why=h_why,line=h_line,head=h_head) if h_v else None)
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
        atoms=p.get('atoms') or []
        st_t,st_h=self._stated(work,atoms); q=self._quantity(p,work,cited)
        used={}
        # techs
        if st_t:
            techs=min(int(st_t['v']),4); why_t=f"stated in the documents: {st_t['why']}"
            if st_t['head']: used['techs']={'head':st_t['head'],'line':st_t['line']}
        elif prof and prof.get('usual_techs') and (prof.get('techs_share') or 0)>=0.75 and prof['kits']>=2: techs=prof['usual_techs']; why_t=f"{prof['key']} usually gets {prof['usual_techs']}"
        else:
            ut={int(k):v for k,v in F['usual_techs'].items()}; techs=max(ut,key=ut.get) if ut else 1; why_t=f'usual crew for {fam} jobs'
        # hours per tech
        n=q['count'] if q and q['unit'] not in ('site','user') and q['p']>=0.5 else None
        if st_h:
            hpt=st_h['v']; why_h=f"stated in the documents: {st_h['why']}"
            if st_h['head']: used['hours_per_tech']={'head':st_h['head'],'line':st_h['line']}
        elif n and F.get('per_unit_crew_hours') and fam in self.b['count_scaled_families'] and n<=(300 if fam in ('ap','camera','power') else 60):
            hpt=max(min(F['median_hours_per_tech'],8),n*F['per_unit_crew_hours']/techs); why_h=f'{n} {q["unit"]}s at the usual time each for {fam} jobs'
        else: hpt=F['median_hours_per_tech']; why_h=f'typical for {fam} jobs'
        hpt=_snap(hpt)
        # rate line and price
        rp=seller or (account if account and account['rate_line_share']>=0.6 else None)
        rate_line=rp['rate_line'] if rp else F['usual_rate_line']; price=rp['price'] if rp else F['usual_price']
        why_r=f"{rp['key']} profile ({rp['kits']} kits)" if rp else f'usual for {fam} jobs'
        # A line the level head marked and that states a level outright sets the rate line's level.
        # The profile and the family decide WHICH rate line; only the level is the head's to say.
        lvl,lvl_line=self._level(atoms,work)
        if lvl and re.match(r'^L[123]\b',str(rate_line)):
            if int(str(rate_line)[1])!=lvl:
                rate_line=f'L{lvl}'+str(rate_line)[2:]
                why_r=f'{why_r}; read:tech_level states L{lvl} on this deal'
            used['rate_line_level']={'head':'read:tech_level','line':lvl_line,'level':lvl}
        alltext=' '.join(a.get('text') or '' for a in work)
        flags=[]
        # The day-type head answers the question the after-hours rule was guessing at. When it has
        # answered anything about this deal's windows, the rule does not run at all: it fires on the
        # word inside an exclusion and on a Sunday completion date, and it cannot tell a window to
        # work in from one to stay out of.
        win=self._window(atoms,work)
        def _where(lab): return '' if win[lab]['in_work'] else ', on a line outside the work section'
        if win:
            off=[lab for lab in OFF_HOURS if lab in win]
            if off:
                top=max(off,key=lambda l:win[l]['p'])
                flags.append('work outside business hours: '+', '.join(lab.replace('_',' ') for lab in off)
                             +f" (read:work_window, p {win[top]['p']:.2f}{_where(top)})")
                used['day_type']={'head':'read:work_window','labels':off,'line':win[top]['line'],
                                  'in_work':win[top]['in_work']}
            if 'blackout' in win:
                flags.append(f"blackout window stated (read:work_window, p {win['blackout']['p']:.2f}"
                             f"{_where('blackout')}): check the dates")
        elif AFTER.search(alltext): flags.append('after-hours work mentioned')
        bill,bill_p,bill_by,bill_line=self._billing(atoms,work,(p.get('heads') or {}).get('deal') or {})
        if bill:
            used['billing']={'head':bill_by,'line':bill_line,'p':bill_p}
        elif bill_by:
            flags.append(f'billing shape unsettled: {bill_by}')
        if HEIGHT.search(alltext): flags.append('height / lift access mentioned')
        if (p.get('site_count') or 0)>=2: flags.append(f"{p['site_count']} places in the documents: check the site list")
        if fam=='program': flags.append('ongoing or multi-site engagement: confirm the duration and site count')
        # Who furnishes the parts: the supplier head, not a wording rule.
        cf=[a.get('id') for a in work if (g:=_said(a,'col:supplier')) and 'customer' in _labels(g[0])]
        if cf:
            flags.append(f'{len(cf)} work line(s) the supplier head reads as customer-furnished')
            used['customer_furnished']={'head':'col:supplier','lines':cf[:12],'n':len(cf)}
        checked=(q['unit'],q['count']) if q and q['p']>=0.5 and q['unit'] not in ('site','user','device') and re.search(r'\b('+UNITS[q['unit']]+r')\b',scope,re.I) else None
        task=self._word(name,work,p.get('c3_brief') or {},checked,p.get('account')) if word else None
        verdict=f"hours={hpt*techs:g};per=job;qty=1;role={rate_line};techs={techs};hpt={hpt:g}"
        return dict(version=self.version,job_type=fam,job_type_p=round(max(sims),3),
            lines=[dict(site='job site',task=task or {'av':'Install AV equipment','camera':'Install cameras','ap':'Install access points','power':'Replace UPS / batteries','rack':'Rack and stack equipment','cabling':'Run network cabling','device':'Install and configure devices','survey':'Site survey and report','program':'Onsite support (staff augmentation)'}[fam],
                techs=techs,hours_per_tech=hpt,crew_hours=hpt*techs,rate_line=rate_line,price=price,why_techs=why_t,why_hours=why_h,why_rate=why_r,proposed_hours=verdict)],
            quantity=({'unit':q['unit'],'count':q['count'],'p':round(q['p'],2),'text':q['text'],'atom_id':q['atom_id']} if q else None),flags=flags,
            billing=({'shape':bill,'p':bill_p,'by':bill_by,'line':bill_line} if bill
                     else ({'shape':None,'p':bill_p,'by':bill_by,'line':bill_line} if bill_by else None)),
            work_window=win or None,
            # Which numbers a super head decided, and which fell back to a profile or a family
            # median. Without this the quote looked equally head-decided either way: 010452 came
            # back with a family median for every number and nothing said so.
            heads={'asked':sorted({h for a in atoms for h in (a.get('reads') or {})}),'used':used,
                   'fell_back':[k for k in ('techs','hours_per_tech','billing','day_type') if k not in used]})
    def _word(self,name,work,brief,checked,account=None):
        jd=[(j.get('text') or '').strip() for j in (brief.get('job_details') or [])][:4]
        facts=f"Deal name: {name}\nJob details: {' | '.join(j[:220] for j in jd) or ' | '.join((a.get('text') or '')[:200] for a in work[:4]) or '(none)'}\n"
        if checked: facts+=f"Checked count: {checked[1]} {checked[0]}\n"
        prompt=("Write ONE Deal Kit labor task line for this field-services job, at most 9 words, starting with a verb. Use only the facts given and describe the work "
                "the customer is asking us to do (not drawings, not other projects). The deal name is the most reliable statement of what the job is. The ONLY number allowed "
                "is the checked count (if one is given, use it exactly); no other digits or number words. No customer, street or city names, no dates, weekdays or months, "
                "no prices, no model or part numbers, no punctuation at the end.\n\n"+facts+"\nTask line:")
        if not self.wording: return None
        try: out=self._post('/api/generate',{'model':self.wording,'prompt':prompt,'stream':False,'think':False,'options':{'temperature':0}})['response'].strip().split('\n')[0]
        except Exception: return None
        out=re.sub(r'^[\-\*"\s]+|["\.\s]+$','',out)
        allowed={str(checked[1])} if checked else set()
        WORDS=r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|single)\b'
        DATES=r'\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|\b(mon|tues|wednes|thurs|fri|satur|sun)day\b'
        def bad(t): return any(n not in allowed for n in re.findall(r'\d[\d,]*',t)) or bool(re.search(DATES,t,re.I)) or (bool(re.search(WORDS,t,re.I)) and not checked) or bool(PART.search(t)) or ' - ' in t
        if bad(out):
            fix=("Rewrite this task line with no digits, sizes, model or part numbers, street names, dates or counts"+(f" except the count {checked[1]}" if checked else "")+
                 ". Under 9 words, start with a verb, plain words a project manager would use, nothing else:\n"+out)
            try: out=re.sub(r'^[\-\*"\s]+|["\.\s]+$','',self._post('/api/generate',{'model':self.wording,'prompt':fix,'stream':False,'think':False,'options':{'temperature':0}})['response'].strip().split('\n')[0])
            except Exception: pass
        out=' '.join(w for w in out.split() if not (re.search(r'\d',w) and re.sub(r'\D','',w) not in allowed))
        # never echo the deal name or the account; a plain line beats a wrong one
        name_words={w.lower() for w in re.findall(r'[A-Za-z]{4,}',(name or '')+' '+(account or '')) if w.lower() not in {'install','installation','camera','cameras','office','support','request','smart','hands','phone','phones','display','displays','battery','cabling','network','rack','switch','survey','room','rooms','site','sites','multi','location','deployment','migration','relocation','replacement','upgrade','refresh'}}
        leaked=[w for w in re.findall(r'[A-Za-z]{4,}',out) if w.lower() in name_words]
        if bad(out) or len(leaked)>=2 or ' - ' in out: return None
        if len(out)>90: out=out[:90].rsplit(' ',1)[0]
        return out or None
