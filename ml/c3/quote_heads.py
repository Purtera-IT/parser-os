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

The heads decide, the parser extracts: when the compile sends the C3 heads' line readings (atom
``reads``: read:co_hours_estimate / read:hours_stated / read:labor_hours, read:crew_size /
read:co_crew_rule, read:tech_level, read:billing_type / jdg:commercial, and the deal answers
jdg:billing_type / jdg:travel_scope), a sure reading picks the line and the number comes off that
line. The EFF / TECHS regex and the family medians answer only when no head is sure, and every number
says which (``by_*``: ``head:<key>`` | ``regex (head unsure)`` | ``regex (heads not asked)`` |
``family median`` ...). Old Deal Kit lines and lines the heads left out are never read.
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
# Height / lift access: a lift named as equipment, or a height stated as a ceiling/mounting height.
# A bare length ("200 foot run", "MDF to the Roof") is cable, not access.
HEIGHT=re.compile(r"\b(?:scissor|boom|aerial|man|personnel)[- ]?lifts?\b|\blifts?\s+(?:rental|access|required|needed|is required)\b|\bbucket truck\b"
                  r"|\b(?:ceilings?|heights?|mount(?:ed|ing)?(?: height)?|high|tall)\b[^.\n]{0,25}?\b\d{2}\s*(?:-\s*\d{2}\s*)?(?:ft|feet|foot|')"
                  r"|\b\d{2}\s*(?:-\s*\d{2}\s*)?(?:ft|feet|foot|')\s*(?:ceilings?|high|tall|clear(?:ance)?|height)\b",re.I)
# After-hours work: the work's own schedule, not a weekend or a night someone mentions in passing.
AFTER=re.compile(r"\bafter[- ]hours\b|\boff[- ]hours\b|\bovernight\b(?!\s+(?:shipping|ship|delivery|express|air|mail))|\bnight (?:shift|work|window)s?\b"
                 r"|\b(?:work|install(?:ation)?|cutover|onsite|on-site)\b[^.\n]{0,30}?\b(?:on|over|during) (?:the )?weekends?\b"
                 r"|\b(?:nights|evenings) and weekends\b|\boutside (?:of )?(?:normal )?business hours\b",re.I)
NOISE=re.compile(r'net\s*\d+|warranty|licen|within \d+ (?:hours|days)|notice|24-48|24\s*/\s*7|quotes? (?:delivered|in)|business days|window',re.I)
HB=(2,4,6,8,10,12,16,24,32,40,48,64,80,120,200,400,600)
SKIP_TYPES={'small_talk','stakeholder','deal_metadata'}
#: A head reading at or above this decides (SOWSMITH_C3_QUOTE_SURE).
SURE=float(os.environ.get('SOWSMITH_C3_QUOTE_SURE','0.8') or 0.8)
#: Hours heads, in the order they decide: the kit's own hours estimate, then hours the documents state.
HOURS_HEADS=('read:co_hours_estimate','read:hours_stated','read:labor_hours')
CREW_HEADS=('read:crew_size','read:co_crew_rule')
BILLING_LINE_HEADS=('read:billing_type','jdg:commercial')
#: Heads that score below live; an answer from one is used but marked (AUDIT_dealkit_heads item 12).
CAVEATS={'read:labor_hours':'read:labor_hours scores below live (0.03-0.04 held-out): check these hours'}
HOURS_ON_LINE=re.compile(r"(\d{1,3}(?:\.\d)?)(?:\s*(?:-|to)\s*(\d{1,3}(?:\.\d)?))?\s*(?:hours?|hrs?|h)\b",re.I)
CREW_ON_LINE=re.compile(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:\(\d+\)\s*)?(?:onsite\s+|field\s+|dedicated\s+)?(?:techs?|technicians?|engineers?|installers?|people|resources?|guys)\b",re.I)
SITES_SAID=re.compile(r"\b(\d{1,4})[ \t]+(?:[\w-]+[ \t]+){0,2}?(?:sites?|locations?|stores?|hotels?|offices?|branches)\b(?![- ]to[- ])",re.I)
#: "5 tanks per site", "10 Site-to-Site tunnels": a count OF something at each site, not a count of sites.
PER_UNIT=re.compile(r"\b(?:per|each|every)\b",re.I)
SITE_TO_SITE=re.compile(r"^[\s-]*to[\s-]",re.I)
#: A word that negates a flag's wording a few words before it ("no lift requirements", "not after hours").
NEGATED=re.compile(r"\b(?:no|not|never|without|excluding|except|n't|none|nor)\b(?:\W+\w+){0,3}\W*$",re.I)
#: Text read off a picture (OCR / vision): room numbers on a floor plan are not counts of anything.
PICTURE_TEXT=re.compile(r"^\s*\[(?:OCR|Image|Figure|vision)",re.I)
LEVEL_ON_LINE=re.compile(r"\b(?:l|level|lvl|tier)\s*-?\s*([1-4])\b",re.I)

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
    @staticmethod
    def _readable(p):
        """The atoms the quote may read: never an old Deal Kit line (a requote would price itself off
        the last kit) and never a line the heads left out of the brief (co_action reject etc.)."""
        out=set((p.get('c3_brief') or {}).get('left_out') or [])
        return [a for a in p.get('atoms') or [] if not a.get('deal_kit') and not a.get('left_out') and a.get('id') not in out]
    @staticmethod
    def _heads_asked(p):
        h=p.get('heads')
        return bool(h.get('asked')) if isinstance(h,dict) else any(a.get('reads') for a in p.get('atoms') or [])
    @staticmethod
    def _sure(a,head):
        r=(a.get('reads') or {}).get(head)
        if not r or r[0] in (None,'_absent','false','none') or float(r[1] or 0)<SURE: return None
        return r
    def _head_number(self,atoms,heads,rx,cap):
        """(value, head, atom id, p) from the surest line a head reads as stating it: the read's value
        when the server sends one, else the number on that line. None when no head is sure."""
        best=None
        for head in heads:
            for a in atoms:
                r=self._sure(a,head)
                if not r: continue
                v=r[2] if len(r)>2 and isinstance(r[2],(int,float)) and r[2]>0 else None
                if v is None:
                    for m in rx.finditer(a.get('text') or ''):
                        g=[x for x in m.groups() if x]; n=[float(NUM.get(x.lower(),x)) for x in g]
                        v=sum(n)/len(n); break
                if v is None or not 0<v<=cap: continue
                if best is None or float(r[1])>best[3]: best=(v,head,a.get('id'),float(r[1]))
            if best: return best
        return None
    def _head_level(self,atoms):
        for a in sorted(atoms,key=lambda a:-float(((a.get('reads') or {}).get('read:tech_level') or [0,0])[1] or 0)):
            r=self._sure(a,'read:tech_level')
            if not r: continue
            m=LEVEL_ON_LINE.search(a.get('text') or '')
            if m: return int(m.group(1)),a.get('id'),float(r[1])
        return None
    def _billing(self,p,atoms):
        d=((p.get('heads') or {}).get('deal') or {})
        for k in ('jdg:billing_type',):
            r=d.get(k)
            if r and r[0] not in (None,'_absent') and r[1] is not None and float(r[1])>=SURE: return dict(answer=r[0],p=round(float(r[1]),3),by=f'heads:{k}',atom_ids=[])
        votes={}
        for a in atoms:
            for h in BILLING_LINE_HEADS:
                r=self._sure(a,h)
                if r: votes.setdefault(r[0],[]).append((a.get('id'),float(r[1]),h))
        if votes:
            v,ev=max(votes.items(),key=lambda kv:(len(kv[1]),max(x[1] for x in kv[1])))
            return dict(answer=v,p=round(max(x[1] for x in ev),3),by=f'heads:{ev[0][2]}',atom_ids=[x[0] for x in ev][:8])
        return dict(answer=None,p=None,by='none',reason='no billing head sure' if self._heads_asked(p) else 'heads not asked',atom_ids=[])
    def _travel(self,p):
        r=((p.get('heads') or {}).get('deal') or {}).get('jdg:travel_scope')
        if r and r[0] not in (None,'_absent') and r[1] is not None and float(r[1])>=SURE: return dict(answer=r[0],p=round(float(r[1]),3),by='heads:jdg:travel_scope')
        return dict(answer=None,p=None,by='none',reason='travel head unsure' if self._heads_asked(p) else 'heads not asked')
    @staticmethod
    def _sites_said(atoms):
        n=0
        for a in atoms:
            t=a.get('text') or ''
            if PICTURE_TEXT.match(t): continue
            for m in SITES_SAID.finditer(t):
                if PER_UNIT.search(m.group(0)): continue
                v=int(m.group(1))
                if v<=2000 and not SIZE.match(t[m.start(1):m.end(1)+7]): n=max(n,v)
        return n
    @staticmethod
    def _said(lines,rx):
        """Ids of the lines whose wording matches ``rx`` and does not negate it."""
        out=[]
        for a in lines:
            t=a.get('text') or ''
            if PICTURE_TEXT.match(t): continue
            if any(not NEGATED.search(t[max(0,m.start()-40):m.start()]) for m in rx.finditer(t)): out.append(a.get('id'))
        return out
    def _work(self,p):
        brief=p.get('c3_brief') or {}; secs=brief.get('sections') or {}
        kept={x.get('id') for x in (secs.get('scope_of_work') or [])}
        cited={c for j in (brief.get('job_details') or []) for c in (j.get('cites') or [])}
        work=[]
        for a in self._readable(p):
            t=(a.get('text') or '').strip(); ty=a.get('type') or a.get('atom_type')
            if not t or ty in SKIP_TYPES: continue
            if a.get('id') in kept or a.get('id') in cited or ty=='deal_summary': work.append(a)
        return work,cited
    def _quantity(self,p,work,cited):
        name=(p.get('deal_name') or '').lower(); groups={}
        roster=int(p.get('site_count') or 0)>0
        wid={a.get('id') for a in work}
        for a in self._readable(p):
            if (a.get('type') or a.get('atom_type')) in SKIP_TYPES: continue
            t=a.get('text') or ''
            found=[]
            for u,rx in PAT.items():
                for m in rx.finditer(t):
                    if SIZE.match(t[m.start(1):m.end(1)+7]): continue
                    if u=='site':
                        # The heads' roster says how many sites; the wording only when there is none.
                        if roster: continue
                        if PICTURE_TEXT.match(t) or SITE_TO_SITE.match(t[m.end():m.end()+6]) or PER_UNIT.search(m.group(0)): continue
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
        emails={m.group(0).lower() for a in self._readable(p) for m in re.finditer(r'[\w.\-]+@[\w\-]+\.(?:com|net|org|io)',a.get('text') or '') if 'purtera' not in m.group(0).lower()}
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
        readable=self._readable(p); asked=self._heads_asked(p)
        unsure='fallback:regex_head_unsure' if asked else 'fallback:regex_heads_not_asked'
        caveats=[]; used={}
        # techs: a sure crew head picks the line, the number comes off it; regex / profile / family only after
        hc=self._head_number(readable,CREW_HEADS,CREW_ON_LINE,20)
        if hc: techs=int(round(hc[0])); why_t=f'stated on the line {hc[1]} reads as the crew'; by_t=f'heads:{hc[1]}'; used['techs']=dict(head=hc[1],atom_id=hc[2],p=round(hc[3],3))
        elif st_t: techs=min(st_t,4); why_t='stated in the documents'; by_t=unsure
        elif prof and prof.get('usual_techs') and (prof.get('techs_share') or 0)>=0.75 and prof['kits']>=2: techs=prof['usual_techs']; why_t=f"{prof['key']} usually gets {prof['usual_techs']}"; by_t='fallback:account_profile'
        else:
            ut={int(k):v for k,v in F['usual_techs'].items()}; techs=max(ut,key=ut.get) if ut else 1; why_t=f'usual crew for {fam} jobs'; by_t='fallback:family_usual'
        # hours per tech
        n=q['count'] if q and q['unit'] not in ('site','user') and q['p']>=0.5 else None
        hh=self._head_number(readable,HOURS_HEADS,HOURS_ON_LINE,2000)
        if hh:
            hpt=hh[0]; why_h=f'stated on the line {hh[1]} reads as the hours'; by_h=f'heads:{hh[1]}'; used['hours']=dict(head=hh[1],atom_id=hh[2],p=round(hh[3],3))
            if hh[1] in CAVEATS: caveats.append(CAVEATS[hh[1]])
        elif st_h: hpt=st_h; why_h='stated in the documents'; by_h=unsure
        elif n and F.get('per_unit_crew_hours') and fam in self.b['count_scaled_families'] and n<=(300 if fam in ('ap','camera','power') else 60):
            hpt=max(min(F['median_hours_per_tech'],8),n*F['per_unit_crew_hours']/techs); why_h=f'{n} {q["unit"]}s at the usual time each for {fam} jobs'; by_h='fallback:family_per_unit_x_regex_count'
        else: hpt=F['median_hours_per_tech']; why_h=f'typical for {fam} jobs'; by_h='fallback:family_median'
        if not by_h.startswith('heads:') and asked: caveats.append('hours not stated on any line the heads read: '+by_h)
        hpt=_snap(hpt)
        # rate line and price
        rp=seller or (account if account and account['rate_line_share']>=0.6 else None)
        rate_line=rp['rate_line'] if rp else F['usual_rate_line']; price=rp['price'] if rp else F['usual_price']
        why_r=f"{rp['key']} profile ({rp['kits']} kits)" if rp else f'usual for {fam} jobs'
        by_r='fallback:account_profile' if rp else 'fallback:family_usual'
        lv=self._head_level(readable)
        if lv:
            m=re.match(r'L\d\s+(.*)',rate_line or '')
            new=f'L{lv[0]} '+(m.group(1) if m else (rate_line or '')).strip()
            if new!=rate_line: caveats.append(f'price {price} is for {rate_line}; check the {new} rate'); why_r=f'level stated on a line read:tech_level reads ({why_r} for the track)'
            rate_line=new; by_r='heads:read:tech_level'; used['level']=dict(head='read:tech_level',atom_id=lv[1],p=round(lv[2],3))
        billing=self._billing(p,readable); travel=self._travel(p)
        flags=[]
        # No day-type or height head yet: these are wording rules, read only on the work lines the heads
        # kept in the brief body (never chatter: 010095 priced overtime off World Cup talk), never when
        # the wording negates them ("no lift requirements", 010228), and marked rule:* with their lines.
        after_ids=self._said(work,AFTER); height_ids=self._said(work,HEIGHT)
        after=bool(after_ids)
        if after: flags.append('after-hours work mentioned (rule:after_hours_text)')
        day_type,by_d=('ot','rule:after_hours_text') if after else ('reg','fallback:default')
        if height_ids: flags.append('height / lift access mentioned (rule:height_text)')
        roster=int(p.get('site_count') or 0)
        if roster>=2: flags.append(f"{roster} places in the documents: check the site list")
        # Site count: the heads' roster when there is one; with none, the deal head that reads a
        # declared site count; the wording only when neither answers, and marked.
        site_by='heads:physical_site' if roster else 'none'
        said=0
        if not roster:
            r=((p.get('heads') or {}).get('deal') or {}).get('deal:declared_site_count')
            if r and r[0] not in (None,'_absent') and r[1] is not None and float(r[1])>=SURE:
                try: said=int(float(r[2] if len(r)>2 and isinstance(r[2],(int,float)) else r[0])); site_by='heads:deal:declared_site_count'
                except (TypeError,ValueError): said=0
            if not said:
                said=self._sites_said(readable)
                if said: site_by='rule:sites_said_text (no roster)'
            if said>=2: flags.append(f"documents mention {said} sites ({site_by}); no site roster from the heads")
        if fam=='program': flags.append('ongoing or multi-site engagement: confirm the duration and site count')
        checked=(q['unit'],q['count']) if q and q['p']>=0.5 and q['unit'] not in ('site','user','device') and re.search(r'\b('+UNITS[q['unit']]+r')\b',scope,re.I) else None
        task=self._word(name,work,p.get('c3_brief') or {},checked,p.get('account')) if word else None
        verdict=f"hours={hpt*techs:g};per=job;qty=1;role={rate_line};techs={techs};hpt={hpt:g}"
        # the atoms each line came from, so a kit edit can become a label on the line that caused it
        src=list(dict.fromkeys([c for j in ((p.get('c3_brief') or {}).get('job_details') or []) for c in (j.get('cites') or [])]
                               +[u['atom_id'] for u in used.values()]+([q['atom_id']] if q else [])))
        src=[x for x in src if x]
        cites=[c for j in ((p.get('c3_brief') or {}).get('job_details') or []) for c in (j.get('cites') or [])]
        by_id={a.get('id'):a for a in readable}
        lead=(used.get('hours') or used.get('techs') or {}).get('atom_id')
        source_text=((by_id.get(lead) or {}).get('text') or '')[:300] if lead else (q['text'] if q else ((work[0].get('text') or '')[:300] if work else ''))
        return dict(version=self.version,job_type=fam,job_type_p=round(max(sims),3),
            heads=dict(asked=asked,used=used,sure=SURE),billing=billing,travel=travel,site_count=p.get('site_count') or 0,after_hours=after,
            sites=dict(count=p.get('site_count') or 0,by=(site_by if 'sites' in p else 'none (not sent)'),
                       said=said or None,rule_atom_ids=dict(after_hours=after_ids,height=height_ids),
                       atom_ids=[i for st in (p.get('sites') or []) for i in (st.get('atom_ids') or [])]),
            lines=[dict(site='job site',task=task or {'av':'Install AV equipment','camera':'Install cameras','ap':'Install access points','power':'Replace UPS / batteries','rack':'Rack and stack equipment','cabling':'Run network cabling','device':'Install and configure devices','survey':'Site survey and report','program':'Onsite support (staff augmentation)'}[fam],
                techs=techs,hours_per_tech=hpt,crew_hours=hpt*techs,rate_line=rate_line,price=price,why_techs=why_t,why_hours=why_h,why_rate=why_r,
                by=dict(techs=by_t,hours=by_h,rate=by_r,day_type=by_d),day_type=day_type,after_hours=after,caveats=caveats,
                atom_ids=src,cites=cites,source_text=source_text,proposed_hours=verdict)],
            quantity=({'unit':q['unit'],'count':q['count'],'p':round(q['p'],2),'text':q['text'],'atom_id':q['atom_id']} if q else None),flags=flags)
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
