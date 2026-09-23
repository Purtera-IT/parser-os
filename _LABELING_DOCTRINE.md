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

## A label may only use what the head will see

The worst thing in my first four labels was not a wrong value. It was two
notes citing facts that appear **nowhere in the documents**: that AJ has
worked with Alec before (Lilli got that by asking him) and that the club
owner is the decision maker. I tagged those labels `domain_knowledge` as if
naming the leak excused it.

It does not. A head is trained on the atom and its context. A label decided
on information that is not in that context teaches the model to assert
things it has no evidence for -- confidently, because the gold said so.

So:

* `domain_knowledge` means **industry** knowledge: what a maglock is, that
  Cat6 outruns Cat5e, that a relay is not a lock. It is knowledge any
  qualified reader brings to the same words.
* It does **not** mean deal history someone told you. If the fact came from
  a conversation, it is not a label -- it is either an atom of its own (if
  someone wrote it down) or it belongs in the cross-deal party index, which
  does not exist yet (PUR-282).
* The test: **could a careful stranger, given only this atom, its neighbours,
  its lead-in and the deal's other documents, reach this label?** If not, the
  label is contaminated, however true it is.

The same rule kills a subtler cheat. "Chase Alec, he has history with us" is
a fine instruction and a terrible label, because the head cannot see history.
"There was a conversation this thread does not contain" is the same finding
stated from the evidence, and it is learnable.

## Structure is a link, not a new atom

`lead_in` already ties "Relay" to "Provided by us:". Nothing tied "Provided
by us:" to the line that opened the whole ask:

```
"Here are the details for the small job I was discussing earlier."   <- announces
  "Provided by us:"            -> Relay, PC, USB cable, converter, extenders
  "Provided by Club/installer:" -> mag lock cable, power supply, lock, Cat5e/6
```

That missing level is why `job_scale: small` was stranded. On its own it is a
fact about one sentence. Linked, it is a fact about **those ten items** --
and Deal Kit no longer has to guess whether "small" covers the whole thread,
which also carries a second question set and another site.

Two pieces, deliberately separate:

* `opens_block` is a **reading** on the announcing line: this line frames what
  follows. One boolean per line, and a human answers it instantly.
* `governs` is a **link** from that line to every atom it covers. It carries
  no text -- which is the whole point. Structure is the one thing the parser
  may assert about other atoms without minting a sentence nobody said.

A leaf gets its meaning back from the chain: "Anything in Orange" is noise;
"Provided by us -> Anything in Orange" is a line; "For the small job ->
provided by us -> anything in orange" is a statement someone can label.

## A conditional is the critical path

> "**If** you all would be able to do something like this, I will get a
> conversation going with the club owner."

Typed `commitment`, this reads as a promise he made. It is the opposite: it
is a promise **waiting on us**. The deal's next move is ours, and nothing in
a type or a `wants` says "blocked on".

`blocked_on: us | customer | partner` says it. It is the difference between a
deal that is waiting for a customer and a deal that is sitting in our own
inbox, and it is the single most useful thing a brief can lead with.

## A picture is content

A drawing is not a decoration on an atom -- it is often the only complete
statement in the document. 010288's door-access diagram says which side the
reader is on, where the RS232 kit sits and who supplies what, and none of
that is written in any sentence.

Three things follow, and each one cost a day to learn:

* **Find it wherever it was written.** The parser could see a picture in one
  shape: a note field whose whole value was a bare link. The same drawing in
  an email body, or written `Diagram: <link> (rev 1)`, produced nothing.
  `app/core/linked_pictures.py` now runs over every atom from every parser.
* **A line carrying a picture is never small talk.** However chatty the words
  around it are, the line is the only pointer to the drawing.
* **The picture rides on the sentence, like every other reading.** It is
  ``image_url`` on the atom whose text names it, plus a ``points_at_artifact``
  read -- not an atom of its own. Same rule as everything else here.

And the deal should end up HOLDING it. `Diagram:` and a link raises a chase
item -- *go and get the drawing we do not hold*. The API copies the picture
into the deal's own storage on first view, which is that chase item being
worked: it renders with no token, and it survives the vendor deleting the
file. A link is a promise; a copy is the artifact.

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
