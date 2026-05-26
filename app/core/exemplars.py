"""Curated exemplar sentences per entity type — drive the embedding-
based retrieval system (v38).

For each entity type we want to extract universally across bid
packages, we provide 8-15 example sentences that cover the major
surface forms PMs / contractors use. The retrieval system computes
cosine similarity between these exemplars and every sentence in the
deal's documents, then keeps sentences above a similarity threshold.

Why this works universally:
  - "The contractor shall provide quarterly reports" and "Vendor will
    submit weekly status updates" both embed to nearly the same point
    in semantic space.
  - We don't need a regex for every requirement-introducing verb
    (shall / must / will / agrees / is required to / covenants /
    warrants / undertakes / commits to). Embeddings handle synonymy.

Curation notes:
  - Examples drawn from real corpora (the 19 packs) — not invented.
  - Each example is a COMPLETE SENTENCE (the retrieval matcher works
    sentence-to-sentence).
  - Mix of verb-led, subject-led, and passive forms.
  - Mix of strict-language (shall / must) and softer (agrees / will).
  - Include a "negative" tone where helpful (e.g. requirement that
    something must NOT be done).
"""

# ════════════════════════════════════════════════════════════════════
# REQUIREMENTS — "things the contractor / vendor / district has to do"
# ════════════════════════════════════════════════════════════════════

REQUIREMENT_EXEMPLARS: list[str] = [
    # Classic shall/must
    "The contractor shall provide quarterly performance reports.",
    "Vendor must comply with PCI-DSS Level 1 security standards.",
    "Contractor shall maintain workers compensation insurance.",
    "The bidder must submit proof of insurance prior to award.",
    # Will / agrees / covenants
    "The contractor agrees to defend and indemnify the district.",
    "Heartland will not utilize a subcontractor without written consent.",
    "Vendor covenants to deliver all materials within 30 days of award.",
    # Required to / is responsible for
    "The contractor is required to furnish a performance bond.",
    "Bidder is responsible for all site safety compliance.",
    # Negative requirements
    "Contractor shall not assign this contract without express written consent.",
    "The vendor must not disclose confidential information.",
    "Contractor shall not offer additional products not required by the customer.",
    # District / customer rights
    "The district may terminate this contract in whole or in part.",
    "Customer reserves the right to inspect all delivered equipment.",
    # Technical compliance
    "The solution shall integrate with the customer's existing identity provider.",
    "The system must support single sign-on via SAML 2.0.",
    "The platform adheres to NIST 800-53 security controls.",
    # Procedural
    "Prior to commencement of the work, the contractor shall furnish a schedule.",
    "Contractor agrees to notify the district within seven business days of any filing.",
]


# ════════════════════════════════════════════════════════════════════
# QUANTITIES — "structural numbers PMs need: SLAs, counts, durations"
# ════════════════════════════════════════════════════════════════════

QUANTITY_EXEMPLARS: list[str] = [
    # SLA / uptime
    "Heartland guarantees 99.999% uptime for all production systems.",
    "The platform maintains 99.95% availability per the service level agreement.",
    "Severity 1 incidents are resolved within 2 hours of report.",
    "Failover to the disaster recovery site completes within 5 minutes.",
    # Counts
    "Approximately 32 schools are served by Beaufort County School District.",
    "The deployment includes 97 wireless access points across three sites.",
    "There are 24 cameras in the proposed VMS configuration.",
    # Help desk / support hours
    "Support is available Monday through Friday from 8 AM to 5 PM Eastern.",
    "The 24/7 monitoring center operates with two engineers per shift.",
    # Payment / commercial terms
    "Net-30 payment terms apply to all invoices.",
    "The customer shall remit payment within 45 days of receipt of invoice.",
    # Contract / warranty durations
    "The contract term is five years with two one-year renewal options.",
    "All hardware carries a three-year manufacturer warranty.",
    "Software maintenance is included for the initial 12 months.",
    # Performance numbers
    "Cisco C9166D1 access points have a 6-8 week lead time at quoted volume.",
    "TSA badging adds 5 to 7 business days lead time per technician.",
]


# ════════════════════════════════════════════════════════════════════
# STAKEHOLDERS — "named human contacts, project roles, signers"
# ════════════════════════════════════════════════════════════════════

STAKEHOLDER_EXEMPLARS: list[str] = [
    # Bid contact lines
    "All questions regarding this RFP should be directed to Kaylee Yinger.",
    "Please contact Lisa Brock at the procurement department for clarification.",
    "Submitted by Michael Panzica, Senior Project Manager.",
    # Email signature blocks
    "Glenn Tilleman, IT Director, glenn.tilleman@hood.k12.tx.us.",
    "Matthew Brener, BRS Inc., (267) 688-7301, matthew@brs.com.",
    # Role-titled introductions
    "The project will be led by Priya Narang, OPTBOT IT Project Lead.",
    "Camila Brooks serves as the Purtera Field Lead for this engagement.",
    "Jordan Ames is the executive sponsor from the customer side.",
    # Sign-off lines
    "Authorized signature: Randall Hughes, Director of Procurement.",
    "This proposal is submitted by John Foster on behalf of Convergent Tech Partners.",
    # Project organization callouts
    "Heartland's implementation team is led by Shaun Tozer.",
    "Renee Watkins will serve as the IT Director for post-go-live support.",
    # NEW v41 — Team roster / signature block patterns
    "Front of the House: Lisa Brock/Implementation Project Manager, Randall Hughes/Specialist, Michael Panzica/Client Support.",
    "Account Team: Alice Brown — Customer Success, Bob White — Solution Architect.",
    "Implementation Team consists of Lisa Brock, Michael Panzica, and Sarah Smith.",
    "Key personnel: Kaylee Yinger (PM), Lisa Brock (Implementation Lead), Michael Panzica (Client Support).",
    "Heartland Project Manager: Shaun Tozer; Lead Engineer: Matthew Brener.",
    "Project staff include Jane Doe, John Smith, and Sarah Johnson.",
    "Senior Project Manager Michael Panzica oversees client onboarding.",
    "Engineering lead: Renee Watkins. Field operations: Camila Brooks. Sales: Noah Patel.",
]


# ════════════════════════════════════════════════════════════════════
# SITES — "physical buildings, codes, addresses, campuses"
# ════════════════════════════════════════════════════════════════════

SITE_EXEMPLARS: list[str] = [
    # Site codes
    "The ATL-HQ-01 facility houses the primary network operations center.",
    "STORE-142 is located in the downtown retail concourse.",
    "MDF-3A serves as the main distribution frame on the third floor.",
    # Named facilities
    "Beaufort Elementary School is the largest site by enrollment.",
    "The Atlanta Headquarters building has four floors of office space.",
    "Airport Logistics Annex requires TSA badging for all technicians.",
    # Addresses
    "The customer site is located at 1200 Peachtree St NE, Atlanta GA.",
    "Services will be delivered at 2900 Mink Point Boulevard, Beaufort SC 29901.",
    # Campus / multi-building
    "The Wesley School campus includes three buildings and a maintenance shed.",
    "Innovation Tower at the corporate campus serves as the headquarters.",
    # Sub-building specifics with site context
    "All wiring closets in the Westside Operations Center require new patch panels.",
]


# ════════════════════════════════════════════════════════════════════
# MONEY — "dollar amounts, fees, prices, insurance limits"
# ════════════════════════════════════════════════════════════════════

MONEY_EXEMPLARS: list[str] = [
    # Bid / quote prices
    "The total submitted bid price is $32,400 for the base contract.",
    "Pricing is $995 per Cisco C9166D1 access point at quoted volume.",
    "Annual DNA Spaces subscription is $84,000 per year.",
    # Insurance limits
    "General liability coverage shall be $1,000,000 per occurrence.",
    "Workers compensation insurance is required at a $500,000 minimum.",
    "Cyber liability coverage of $5,000,000 aggregate is required.",
    # Bonds / escrows
    "A performance bond in the amount of 10% of the contract value is required.",
    "Bid security in the form of a $10,000 certified check must accompany the proposal.",
    # Fees
    "A monthly service fee of $2,500 covers managed support.",
    "Late payment penalty is 1.5% per month on outstanding balances.",
    # Capex breakdown
    "The hardware portion of the bill of materials totals $1,432,000.",
    "Service labor is quoted at $185 per hour for installation work.",
]


# ════════════════════════════════════════════════════════════════════
# DATES — "deadlines, go-live, board meetings, milestones"
# ════════════════════════════════════════════════════════════════════

DATE_EXEMPLARS: list[str] = [
    # Submission / award deadlines
    "Proposals are due no later than September 30, 2025 at 3:00 PM EST.",
    "The deadline for written questions is October 6, 2025.",
    "Contract award is anticipated by November 15, 2025.",
    # Go-live / cutover
    "The system shall be operational by July 1, 2026.",
    "Cutover is scheduled for the weekend of July 27-28, 2026.",
    "Production go-live must precede the Q3 board meeting on August 14, 2026.",
    # Phase boundaries
    "Phase 1 software installation begins June 1, 2026 and runs through June 26.",
    "Hardware staging will be complete by 2026-06-26.",
    # Recurring / fiscal
    "Quarterly review meetings are held on the first Tuesday of each quarter.",
    "Annual budget renewal is due by July 1 of each fiscal year.",
    # Notice periods
    "Either party may terminate with 90 days written notice.",
    "Insurance certificates must be provided within 7 days of contract execution.",
]


# ════════════════════════════════════════════════════════════════════
# MILESTONES — "named project phases / gates / deliverables"
# ════════════════════════════════════════════════════════════════════

MILESTONE_EXEMPLARS: list[str] = [
    # Implementation phases
    "Phase 1: Software Installation, Configuration & Testing.",
    "Phase 2: Hardware deployment and site survey completion.",
    "Phase 3: User training and acceptance testing.",
    "Phase 4: Production cutover and go-live.",
    "Phase 5: Post-go-live support and stabilization period.",
    # Gates / approvals
    "Customer acceptance of the design package is a key milestone.",
    "Substantial completion is achieved when all sites pass functional testing.",
    "Final acceptance occurs after the 30-day post-go-live observation period.",
    # Deliverables
    "Submission of the implementation plan is due within 10 days of award.",
    "Training materials must be delivered prior to user acceptance testing.",
    "As-built drawings shall be provided within 30 days of project closure.",
]


# ════════════════════════════════════════════════════════════════════
# RISKS — "things that could go wrong, dependencies, contingencies"
# ════════════════════════════════════════════════════════════════════

RISK_EXEMPLARS: list[str] = [
    # Schedule / lead time
    "Cisco access point lead times may extend the implementation schedule.",
    "TSA badging delays could impact the Airport Annex installation window.",
    # Technical / compatibility
    "Existing switch PoE budget may be insufficient for all proposed APs at full load.",
    "Integration with the legacy PowerSchool import may require custom mapping.",
    # Commercial / contractual
    "Net payment terms disagreement between RFP and signed quote creates ambiguity.",
    "The customer SLA specifies higher uptime than the standard Purtera offering.",
    # Compliance / regulatory
    "HIPAA and PCI dual-regulated workloads on shared infrastructure require legal review.",
    "Subcontractor procurement for electrical work may extend the project timeline.",
    # Operational
    "Cutover window constraints limit installation to after 18:00 ET on weekdays.",
    "Single points of failure in the proposed topology should be evaluated.",
]


# ════════════════════════════════════════════════════════════════════
# NEGATIVE EXEMPLARS — "what each entity type is NOT"
# ════════════════════════════════════════════════════════════════════
#
# Used by v39 margin-scoring: final_score = positive_sim − 0.7 *
# negative_sim. A sentence semantically close to a "what NOT to match"
# exemplar gets penalized, even if it weakly matches positive exemplars.
# Drops product marketing, table headers, generic boilerplate.

REQUIREMENT_NEGATIVE_EXEMPLARS: list[str] = [
    # Product marketing copy
    "Mosaic is a cloud-based platform that helps districts manage food service.",
    "Heartland has been serving school districts since 1997.",
    "Our solution is the industry leader in payment processing.",
    "MySchoolBucks allows parents to securely pay for school meals.",
    "The system provides intuitive dashboards and reporting tools.",
    # Background / history
    "In 2016, Global Payments completed its merger with Heartland Payment Systems.",
    "PCI-DSS is the security standard for payment card data.",
    "The company has more than 100,000 active clients worldwide.",
    # Section headings
    "5.0 Information for Offerors to Submit",
    "Section 7.1.14 Information Security Definitions",
    "Table of Contents",
    "Executive Summary",
    # Table cell fragments
    "Score (/30): 30 (100%)",
    "Bidders Submitted Price: $32,400.00",
    "Page 12 of 156",
    # Generic facts
    "The Mosaic platform adheres to industry standards.",
    "The system is fully encrypted.",
]

STAKEHOLDER_NEGATIVE_EXEMPLARS: list[str] = [
    # Organization names that look like person names
    "The Beaufort County School District manages 32 schools.",
    "Heartland Payment Systems is the contractor.",
    "Customer Support is available Monday through Friday.",
    "End Users may contact support for assistance.",
    "Mosaic Front Office provides administrative functions.",
    # Role-only mentions
    "The project manager will coordinate the implementation.",
    "All technicians must have appropriate badging.",
    "The signing authority is the district superintendent.",
    # Department / function names
    "Information Technology is responsible for system maintenance.",
    "Procurement Department handles all vendor relationships.",
    "Field Operations will conduct site surveys.",
    # Bullet headings
    "Stakeholder Roles and Responsibilities",
    "Contact Information for Project Team",
]

SITE_NEGATIVE_EXEMPLARS: list[str] = [
    # Standards / specifications (NOT physical sites)
    "Compliance with NFPA 70 fire code is required.",
    "All work shall meet IEEE 802.11ax standards.",
    "ANSI/TIA-568 cabling standards apply.",
    "ISO 27001 certification is mandatory.",
    # Vendor / product names misread as sites
    "Microsoft SQL Server manages the database backend.",
    "Cisco Catalyst 9300 switches provide access layer connectivity.",
    "Mosaic Cloud is the hosting environment.",
    "Power School integration is required.",
    # Generic location terms
    "All sites must have power and network connectivity.",
    "The customer site shall be ready for installation.",
    "Each location requires its own configuration.",
    # Form / spec labels
    "Site Code Field",
    "Location of Service Delivery",
    "Address Information",
]

QUANTITY_NEGATIVE_EXEMPLARS: list[str] = [
    # Page numbers / section numbers
    "Page 47 of 156",
    "Section 5.3.2 covers insurance requirements",
    "Item 12 in the bill of materials",
    # Years alone (no quantity context)
    "The contract was signed in 2024.",
    "Established in 1997, Heartland has decades of experience.",
    # Version numbers
    "Mosaic version 7.2 is the current release.",
    "Windows Server 2022 is the target platform.",
    # Phone numbers / IDs
    "Call (800) 555-1234 for support.",
    "Tax ID: 12-3456789",
    # Vague counts
    "Many districts use our solution.",
    "Several years of experience.",
]

MONEY_NEGATIVE_EXEMPLARS: list[str] = [
    # Year alone
    "In 2024, revenue grew significantly.",
    # Phone-number-shaped numerics
    "Reach customer service at 1-800-555-0199.",
    # Population / count (not money)
    "Beaufort County has approximately 192,000 residents.",
    # Standards / version numbers
    "ISO 9001:2015 quality management system.",
    "Section 1.4 of the proposal.",
]

DATE_NEGATIVE_EXEMPLARS: list[str] = [
    # Standard refs / version numbers that contain dates
    "NIST 800-53 revision 5 controls.",
    "ISO 27001:2013 is the prior version.",
    # Historical facts (not actionable dates)
    "Founded in 1997, Heartland is a leader in payments.",
    "The company has been in business for over 25 years.",
    # Phone number IDs (date-like)
    "Account ID: 2024-A-1138",
    # Generic temporal references
    "We have years of experience in this field.",
    "Many years of operational excellence.",
]


# ════════════════════════════════════════════════════════════════════
# REGISTRY — by entity type name
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# v43 NEW ENTITY TYPES — certifications / acceptance / penalties /
# compliance obligations (risks already defined above).
# ════════════════════════════════════════════════════════════════════

CERTIFICATION_EXEMPLARS: list[str] = [
    # Payment / security certs
    "Heartland is certified to PCI-DSS Level 1 at the Service Provider tier.",
    "The vendor maintains SOC 2 Type II certification.",
    "All systems comply with HIPAA Privacy and Security Rules.",
    "FedRAMP Moderate authorization is required for the cloud platform.",
    # Quality / process certs
    "The platform is ISO 27001:2013 certified for information security management.",
    "Operations are audited annually for SSAE 18 SOC 1 compliance.",
    "NIST 800-53 controls are implemented across the production environment.",
    # Education / govt certs
    "USDA approves the menu-planning module for nutrient analysis compliance.",
    "The submission includes a signed FNS-742 application form.",
    "FERPA student-records confidentiality applies to all integration data.",
    # Privacy regulations
    "GDPR data-subject rights are honored for EU resident records.",
    "CCPA opt-out workflows are implemented for California residents.",
    # Industry standards
    "All cabling shall comply with TIA-568-C.2 Category 6A standards.",
    "Wireless deployments meet IEEE 802.11ax (Wi-Fi 6) specifications.",
    "Fire systems are certified to NFPA 72 requirements.",
]

ACCEPTANCE_EXEMPLARS: list[str] = [
    # Acceptance gates
    "Substantial completion is achieved when all sites pass functional testing.",
    "Final acceptance occurs after the 30-day post-go-live observation period.",
    "Customer acceptance of the design package is a prerequisite to procurement.",
    # Deliverables
    "As-built drawings shall be provided within 30 days of project closure.",
    "Test reports for all CAT6A links shall be submitted prior to acceptance.",
    "Training records and user acceptance signoffs are required before go-live.",
    # Quality gates
    "All deliverables must pass third-party penetration testing before acceptance.",
    "Each phase requires customer sign-off before the next phase begins.",
    "Acceptance testing covers functional, performance, and security criteria.",
    # Closeout
    "Closeout documentation includes warranty registrations and maintenance contacts.",
    "Final invoice may be issued only after written acceptance is received.",
]

PENALTY_EXEMPLARS: list[str] = [
    # Service credit / SLA penalties
    "Service credits of 10% of monthly fee apply per hour of unplanned downtime beyond 99.9% uptime.",
    "Late delivery penalty is 1% of contract value per business day overdue.",
    "Failure to meet response-time SLA results in proportional service credits.",
    # Late payment / termination triggers
    "Interest of 1.5% per month accrues on unpaid invoices beyond 30 days.",
    "The District may terminate for default if the cure period of 19 days lapses.",
    "Liquidated damages of $500 per day apply for delays past the agreed milestone.",
    # Material breach
    "Material breach by the contractor entitles the District to immediate termination.",
    "Repeated SLA failures within a quarter trigger contract review.",
    "Any data breach incurs notification within 72 hours and remediation costs.",
    # Bond forfeiture
    "Performance bond may be forfeited upon failure to complete the work.",
    "Bid security is forfeited if the awarded bidder declines the contract.",
]

COMPLIANCE_EXEMPLARS: list[str] = [
    # Regulatory references (not always "shall" but mandatory)
    "All workplace conditions adhere to the Fair Labor Standards Act.",
    "The contractor complies with applicable provisions of the Americans with Disabilities Act.",
    "Equal Employment Opportunity laws apply to all hiring decisions.",
    "Davis-Bacon wage rates apply for all federally-funded labor.",
    # Statute / code references
    "Conformance to South Carolina Code Section 11-35 is required for procurement.",
    "Federal Acquisition Regulation Part 52 clauses are incorporated by reference.",
    "Section 504 of the Rehabilitation Act prohibits discrimination on the basis of disability.",
    "All work conforms to the most current edition of the National Electrical Code.",
    # State-specific
    "South Carolina sales tax exemption applies under SC Code Section 12-36.",
    "California Education Code Section 49073 governs student data privacy.",
    "Texas Government Code 2252 applies to all state-procurement contracts.",
]


EXEMPLARS_BY_TYPE: dict[str, list[str]] = {
    "requirement": REQUIREMENT_EXEMPLARS,
    "quantity": QUANTITY_EXEMPLARS,
    "stakeholder": STAKEHOLDER_EXEMPLARS,
    "site": SITE_EXEMPLARS,
    "money": MONEY_EXEMPLARS,
    "date": DATE_EXEMPLARS,
    "milestone": MILESTONE_EXEMPLARS,
    "risk": RISK_EXEMPLARS,
    # v43 new
    "certification": CERTIFICATION_EXEMPLARS,
    "acceptance": ACCEPTANCE_EXEMPLARS,
    "penalty": PENALTY_EXEMPLARS,
    "compliance_obligation": COMPLIANCE_EXEMPLARS,
}

NEGATIVE_EXEMPLARS_BY_TYPE: dict[str, list[str]] = {
    "requirement": REQUIREMENT_NEGATIVE_EXEMPLARS,
    "stakeholder": STAKEHOLDER_NEGATIVE_EXEMPLARS,
    "site": SITE_NEGATIVE_EXEMPLARS,
    "quantity": QUANTITY_NEGATIVE_EXEMPLARS,
    "money": MONEY_NEGATIVE_EXEMPLARS,
    "date": DATE_NEGATIVE_EXEMPLARS,
}


__all__ = [
    "REQUIREMENT_EXEMPLARS",
    "QUANTITY_EXEMPLARS",
    "STAKEHOLDER_EXEMPLARS",
    "SITE_EXEMPLARS",
    "MONEY_EXEMPLARS",
    "DATE_EXEMPLARS",
    "MILESTONE_EXEMPLARS",
    "RISK_EXEMPLARS",
    "REQUIREMENT_NEGATIVE_EXEMPLARS",
    "STAKEHOLDER_NEGATIVE_EXEMPLARS",
    "SITE_NEGATIVE_EXEMPLARS",
    "QUANTITY_NEGATIVE_EXEMPLARS",
    "MONEY_NEGATIVE_EXEMPLARS",
    "DATE_NEGATIVE_EXEMPLARS",
    "EXEMPLARS_BY_TYPE",
    "NEGATIVE_EXEMPLARS_BY_TYPE",
]
