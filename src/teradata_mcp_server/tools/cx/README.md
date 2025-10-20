## Tools

### `financial_rag_analysis`
Retrieves context from multi-year financial reports and returns evidence for LLM answers.

- Intelligent retrieval
  - Multi-year: balanced chunks per requested year
  - Single-year or global: top results by semantic similarity
- Outputs clean fields for citations: text, doc name, report year, section title, chunk number, similarity
- Enforces answering rules: use only retrieved chunks, include year and document in citations, fail gracefully if not enough context

**Workflow**
1. Store the user question with metadata (years, analysis type).  
2. Tokenize and embed the question with IVSM.  
3. Convert to vector columns.  
4. Run semantic search with optional year filters and per-year balancing.  
5. Return chunks and retrieval metadata.

**Typical queries**
- Temporal: trends across multiple years.  
- Comparative: side-by-side years.  
- General: facts from a single report or globally.

---

### `customer_meetingPrep`
Consolidates customer conversations and profile data to prepare for meetings.

- Sources: phone transcripts, video meetings, email threads, chat logs, plus customer profile
- Adds derived signals: recency buckets, sentiment categories, priority levels, contract urgency
- Supports multiple briefing styles:
  - General briefing: relationship health, talking points, agenda
  - Specific inquiry: quote-and-cite by date and interaction type
  - Timeline or sentiment questions: focus on dates, sequences, and trends

**Workflow**
1. Resolve the customer name with fuzzy matching.  
2. Pull interactions within lookback window.  
3. Rank by priority and recency.  
4. Return structured records ready for analysis and agenda building.

---

## Configuration

All settings are managed in **`financial_rag_config.yaml`**.

You can adjust:
- Database and table locations for questions, models, and vector store  
- Model identifier and basic embedding parameters  
- Retrieval behavior: default and max K, per-year balancing, similarity threshold  
- Vector store schema expectations: required text column and metadata fields  

Keep config as the single source of truth. The tools read it on load and adapt automatically.

---

## Usage Flow

1) For financial report questions  
- Parse user intent and years.  
- Run **`financial_rag_analysis`**.  
- Answer strictly from returned chunks with year and document citations.

2) For customer meeting prep  
- Provide customer name and optional meeting type and lookback window.  
- Run **`customer_meetingPrep`**.  
- Build a briefing or direct answers using the returned, ranked interactions.

---

## Output Guarantees

- JSON responses with `results` and rich `metadata` for traceability.  
- Types converted to JSON-friendly formats.  
- Safe defaults when config files are missing.

---

[Return to Main README](../../../../README.md)
