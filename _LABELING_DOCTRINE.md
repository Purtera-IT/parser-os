# How to think about an atom

This is the thinking behind the labels, written down so it is not re-derived
every deal and not lost in a note field. It came out of labeling deal 010288
(access control for Huzzard, CDW) line by line.

The one rule everything else follows from:

> **An atom is a statement someone made. Not a line of text.**

A line torn off its label, its speaker and its message is a fragment, and
nobody — human or head — can label a fragment. "Relay" means nothing;
"Provided by us → Relay, said by CDW to us" is a fact.

## What an atom is, and what it is not

An atom is **a statement someone made**, anchored where they said it: a span,
a speaker, a document, a line.

What an atom MEANS is not another atom. "The sender calls this a small job"
was minted as a sibling of the sentence it came from -- a card with no
speaker, no line and nothing in the document to highlight, and a labeler
correctly asked what it was. The fact was right; the atom was invented.

So a reading rides on the sentence it was read out of, as ``reads``:

```
atom   "Here are the details for the small job I was discussing earlier."
reads  job_scale = small          (why: "small job")
       points_at_artifact = none
wants  chase-conversation
```

This is also the shape a head has to learn: **a head reads an atom and says
what it implies.** Today a regex fills `reads` -- that is scaffolding, and it
exists only to put a first label in front of a human. A head trained on those
labels answers from meaning, so "a quick one-door install" lands where no
pattern of mine would have.

Deal Kit consumes the reading, not the prose: a small job changes how many PMs
it plans for, which rate card applies, how much site time to assume.

**There is no exception.** I first kept "go and get the diagram we do not
hold" as a row of its own, on the grounds that work is not a restatement. It
is still a sentence nobody said, and on the card it read as one more invented
atom. Work is a READING too -- `chase: conversation`, `needs_artifact:
diagram` -- carried by the line that points at the missing thing. The deal's
chase list is every atom carrying one, which is a view, not a new atom.

Nothing the parser writes is an atom. If the parser has something to say, it
says it about a sentence someone said.

## The three questions

A type alone cannot carry what a PM knows when they read a sentence. Three
questions, three axes:

### 1. What is it? (`type`)

The existing registry: `scope_item`, `commitment`, `open_question`,
`physical_site`, `deal_metadata`, and the rest. This is the conclusion.

### 2. What is it about? (`about`)

| value | means | example from 010288 |
| --- | --- | --- |
| `deal` | this job, this quote | "The club/installer will need to source anything beyond the Relay" |
| `account` | the customer relationship, beyond this job | "it could lead to many more of the same opportunity" |
| `partner` | how a partner behaves with us | "the diagram is labeled by the vendor and the Installer Supplied Components are not accurate" |
| `internal` | how we work, not what we deliver | "Sending it over to my solutions team" |

Why it matters: `deal` reaches the SOW. `account` and `partner` never do —
but they are exactly what belongs in the brief's *why you should care*, and
throwing them away is what made the parser look blind. A PM should never be
asked to treat them as scope, and should never lose them either.

### 3. What does it want from us? (`wants`)

| value | means | example |
| --- | --- | --- |
| `nothing` | it is just true | "They are intending to use a maglock." |
| `chase-artifact` | something exists that we do not hold | "Diagram:" and a link |
| `chase-conversation` | it points at a talk that is not in the thread | "the small job **I was discussing earlier**" |
| `confirm-with-customer` | we must ask before we can price it | "Has the door been installed with the lock?" |
| `decide-internally` | we must decide, nobody else | "We can use only the connectivity kit, or the kit plus the adapters" |

This axis turns an instinct — *there is something we are missing, go find it* —
into a list someone can work.

## Worked examples

**"If you all would be able to do something like this, I will get a
conversation going with the club owner."**
`commitment` · `about: account` · `wants: decide-internally`
A promise, not banter. It is conditional on OUR answer, which makes it a task
on us before it is one on him — and the club owner is a decision maker we have
never spoken to. The parser first filed this as small talk: the worst kind of
failure, because the card disappears instead of being wrong in public.

**"If this is a successful implementation, it could lead to many more of the
same opportunity."**
`deal_metadata` · `about: account` · `wants: nothing`
Says nothing about what we build or who supplies it, so it must not reach the
SOW. It is why a small job matters commercially. Keep it, show it in the
brief, never ask a PM to treat it as scope.

**"Here are the details for the small job I was discussing earlier."**
`deal_metadata` · `about: deal` · `wants: chase-conversation`
Three facts in a throwaway line: the job is **small** (sizing), the detail
**follows below** (everything under it is one ask), and there was a **call we
do not have**. The last one is a gap nobody would notice by reading the
thread.

**"Relay" under "Provided by us:"**
`bom_line` · `about: deal` · `wants: nothing` · supplier: us
Only means anything under its label. Show the label, always.

## What follows from this

* **A rule may predict; only a human may hide.** Small talk is a judgement,
  so the rule leaves `reads: small_talk` on the card and nothing disappears
  from the queue until a labeler says `small_talk`. The regex that used to
  decide hid a promise naming the deal's decision maker, and no one would
  have known: a hidden atom leaves no trace, a wrong type is visible.
* **Never filter before extracting.** Read the facts out of a sentence first;
  only what is left over is small talk. We lost the job size, the diagram and
  the account signal by filtering first.
* **A miss must be visible.** A line that produced no atom is reported
  (`coverage.text`), because a recall failure otherwise leaves no trace and is
  found only by a human reading the source beside the output.
* **Guessing is allowed; guessing silently is not.** A cross-message answer,
  a company's role, a proposed type: propose it, mark it, let a human confirm.
* **A chase item must not send someone hunting where the answer cannot be.**
  "the small job I was discussing earlier" came from a rep AJ had dealt with
  before, so the call predates the deal: the item says to ask him, not to
  search the thread. Knowing WHO we have history with needs a cross-deal party
  index we do not have yet (crm_deal_contacts is empty), so for now it is read
  from the words ("as always", "like last time", "discussing earlier").
* **A judgement a rule cannot make is a head, not a bigger regex.** `about`
  and `wants` are judgements. They ship as labels first, become heads when
  there are enough of them.

## Every rule here is a head waiting for labels

The readings on an atom are heads, one per key, listed in
`app/core/atom_types.json` under `reads`. A regex fills each one TODAY at low
confidence with `source: "rule"`, for one purpose: to put a guess in front of
a labeler. Confirmations and drops are the training set; the head replaces the
rule, and the rule stops deciding anything.

| head | fills today | why a keyword list cannot hold it |
| --- | --- | --- |
| `small_talk` | pleasantry/handoff patterns | it hid "I will get a conversation going with the club owner" -- the deal's decision maker -- because the words looked like banter |
| `job_scale` | "small job", "quick install" | "a one-door retrofit" and "nothing fancy" say the same thing and match nothing |
| `expansion` | "lead to many more" | the same fact arrives as "they have 40 clubs" |
| `commitment` | "I will …" | a promise can be a question ("want me to call the owner?") |
| `introduces_party` | role words after "call/conversation with" | the role is often implied: "I'll talk to Gary, he signs" |
| `chase` | "as discussed", "see attached" | whether it is missing depends on what the deal holds |
| `needs_artifact` | a picture word plus a link | "send the drawing over" names no file type |

A drop is worth as much as a confirmation: a rule can only produce positives,
so "no, that is not small talk" is a label only a human can leave, and it is
what stops the head inheriting the pattern's blind spots.

## Heads this implies (not yet built)

| head | decides | why a rule cannot |
| --- | --- | --- |
| `small_talk` | is this relationship talk or a statement about the work? | it hid "I will get a conversation going with the club owner" -- the deal's decision maker -- on the phrase "get a conversation going" |
| `about` | deal / account / partner / internal | needs to know what the job is |
| `wants` | the five values above | depends on what the deal already holds |
| dangling reference | "as discussed", "attached", "the diagram" → is it here? | the phrase is easy, the resolution is not |
| speaker role | is cdw.com the reseller, the installer, the manufacturer? | it changes per deal |
| party history | have we worked with this person before, and on what? | needs an index across deals, not this one |

Related: `_HEAD_LEDGER.md` (what is trained and what is not).
