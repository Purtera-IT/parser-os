# How to think about an atom

This is the thinking behind the labels, written down so it is not re-derived
every deal and not lost in a note field. It came out of labeling deal 010288
(access control for Huzzard, CDW) line by line.

The one rule everything else follows from:

> **An atom is a statement someone made. Not a line of text.**

A line torn off its label, its speaker and its message is a fragment, and
nobody — human or head — can label a fragment. "Relay" means nothing;
"Provided by us → Relay, said by CDW to us" is a fact.

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

* **Never filter before extracting.** Read the facts out of a sentence first;
  only what is left over is small talk. We lost the job size, the diagram and
  the account signal by filtering first.
* **A miss must be visible.** A line that produced no atom is reported
  (`coverage.text`), because a recall failure otherwise leaves no trace and is
  found only by a human reading the source beside the output.
* **Guessing is allowed; guessing silently is not.** A cross-message answer,
  a company's role, a proposed type: propose it, mark it, let a human confirm.
* **A judgement a rule cannot make is a head, not a bigger regex.** `about`
  and `wants` are judgements. They ship as labels first, become heads when
  there are enough of them.

## Heads this implies (not yet built)

| head | decides | why a rule cannot |
| --- | --- | --- |
| `about` | deal / account / partner / internal | needs to know what the job is |
| `wants` | the five values above | depends on what the deal already holds |
| dangling reference | "as discussed", "attached", "the diagram" → is it here? | the phrase is easy, the resolution is not |
| speaker role | is cdw.com the reseller, the installer, the manufacturer? | it changes per deal |

Related: `_HEAD_LEDGER.md` (what is trained and what is not).
