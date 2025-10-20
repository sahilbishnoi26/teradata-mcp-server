######################################################################################
# Financial Report Analysis Tool
######################################################################################


import logging
import yaml
from typing import Optional, Any, Dict, List
import json
from datetime import date, datetime
from decimal import Decimal
from teradatasql import TeradataConnection
from pathlib import Path

logger = logging.getLogger("teradata_mcp_server")


def serialize_teradata_types(obj: Any) -> Any:
    """Convert Teradata-specific types to JSON serializable formats"""
    if isinstance(obj, date | datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)

def rows_to_json(cursor_description: Any, rows: list[Any]) -> list[dict[str, Any]]:
    """Convert database rows to JSON objects using column names as keys"""
    if not cursor_description or not rows:
        return []

    columns = [col[0] for col in cursor_description]
    return [
        {
            col: serialize_teradata_types(value)
            for col, value in zip(columns, row)
        }
        for row in rows
    ]

def create_response(data: Any, metadata: dict[str, Any] | None = None) -> str:
    """Create a standardized JSON response structure"""
    if metadata:
        response = {
            "status": "success",
            "metadata": metadata,
            "results": data
        }
    else:
        response = {
            "status": "success",
            "results": data
        }

    return json.dumps(response, default=serialize_teradata_types)


def load_financial_rag_config():
    """Load Financial RAG configuration from financial_rag_config.yml"""
    try:
        # Get the directory path
        current_dir = Path(__file__).parent
        # Go to config/
        config_path = current_dir.parent.parent / 'config' / 'financial_rag_config.yml'
        
        with open(config_path, 'r') as file:
            logger.info(f"Loading Financial RAG config from: {config_path}")
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning(f"Financial RAG config file not found: {config_path}, using defaults")
        return get_default_financial_rag_config()
    except Exception as e:
        logger.error(f"Error loading Financial RAG config: {e}")
        return get_default_financial_rag_config()

def get_default_financial_rag_config():
    """Default Financial RAG configuration as fallback"""
    return {
        'version': 'ivsm',
        'databases': {
            'query_db': 'demo_db',
            'model_db': 'demo_db', 
            'vector_db': 'demo_db'
        },
        'tables': {
            'query_table': 'financial_user_query',
            'query_embedding_store': 'financial_user_query_embeddings',
            'vector_table': 'financial_reports_multi_year_embeddings_store',
            'model_table': 'embeddings_models',
            'tokenizer_table': 'embeddings_tokenizers'
        },
        'model': {
            'model_id': 'bge-small-en-v1.5'
        },
        'retrieval': {
            'default_k_per_year': 5,
            'max_k_per_year': 15,
            'default_k_global': 20,
            'max_k_global': 100,
            'min_similarity_threshold': 0.6
        },
        'vector_store_schema': {
            'required_fields': ['txt'],
            'metadata_fields_in_vector_store': ['doc_name', 'report_year', 'section_title', 'chunk_num']
        },
        'embedding': {
            'vector_length': 384,
            'vector_column_prefix': 'emb_',
            'distance_measure': 'cosine',
            'feature_columns': '[emb_0:emb_383]'
        }
    }

# Load config at module level
FINANCIAL_RAG_CONFIG = load_financial_rag_config()

def build_financial_search_query(vector_db, dst_table, chunk_embed_table, k, years, config):
    """Build dynamic search query with optional year filtering and per-year balancing"""
    metadata_fields = config['vector_store_schema']['metadata_fields_in_vector_store']
    feature_columns = config['embedding']['feature_columns']
    min_similarity = config['retrieval'].get('min_similarity_threshold', 0.6)
    
    # Build SELECT clause dynamically
    select_fields = ["e_ref.txt AS reference_txt"]
    
    # Add all metadata fields from vector store
    for field in metadata_fields:
        if field != 'txt':
            select_fields.append(f"e_ref.{field} AS {field}")
    
    # Add similarity
    select_fields.append("(1.0 - dt.distance) AS similarity")
    
    select_clause = ",\n            ".join(select_fields)
    
    # Build WHERE clause
    where_conditions = [f"(1.0 - dt.distance) >= {min_similarity}"]
    
    # Add year filtering if years specified
    if years:
        year_list = ','.join(map(str, years))
        where_conditions.append(f"e_ref.report_year IN ({year_list})")
    
    where_clause = " AND ".join(where_conditions)
    
    # Multi-year strategy
    if years and len(years) > 1:
        k_per_year = config['retrieval']['default_k_per_year']
        total_k = k_per_year * len(years)
        
        query = f"""
        WITH ranked_results AS (
            SELECT
                {select_clause},
                ROW_NUMBER() OVER (PARTITION BY e_ref.report_year ORDER BY (1.0 - dt.distance) DESC) as year_rank
            FROM TD_VECTORDISTANCE (
                    ON {vector_db}.{dst_table} AS TargetTable
                    ON {vector_db}.{chunk_embed_table} AS ReferenceTable DIMENSION
                    USING
                        TargetIDColumn('id')
                        TargetFeatureColumns('{feature_columns}')
                        RefIDColumn('id')
                        RefFeatureColumns('{feature_columns}')
                        DistanceMeasure('cosine')
                        TopK({total_k})
                ) AS dt
            JOIN {vector_db}.{chunk_embed_table} e_ref
              ON e_ref.id = dt.reference_id
            WHERE {where_clause}
        )
        SELECT 
            reference_txt, doc_name, report_year, section_title, chunk_num, similarity
        FROM ranked_results 
        WHERE year_rank <= {k_per_year}
        ORDER BY report_year, similarity DESC
        """
    else:
        # Single year or global query
        query = f"""
        SELECT {select_clause}
        FROM TD_VECTORDISTANCE (
                ON {vector_db}.{dst_table} AS TargetTable
                ON {vector_db}.{chunk_embed_table} AS ReferenceTable DIMENSION
                USING
                    TargetIDColumn('id')
                    TargetFeatureColumns('{feature_columns}')
                    RefIDColumn('id')
                    RefFeatureColumns('{feature_columns}')
                    DistanceMeasure('cosine')
                    TopK({k})
            ) AS dt
        JOIN {vector_db}.{chunk_embed_table} e_ref
          ON e_ref.id = dt.reference_id
        WHERE {where_clause}
        ORDER BY similarity DESC
        """
    
    return query

def handle_Financial_Rag_Analysis(
    conn: TeradataConnection,
    question: str,
    years: List[int] = None,
    analysis_type: str = "general",
    k: int = None,
    *args,
    **kwargs,
):
    """
        Execute financial reports RAG analysis with intelligent multi-year retrieval.
        
        This tool performs financial analysis by retrieving relevant chunks from annual reports
        across one or more years based on LLM-parsed parameters from the user's question.

        INTELLIGENT RETRIEVAL STRATEGIES:
        - MULTI-YEAR ANALYSIS: Gets {default_k_per_year} chunks per year for balanced temporal analysis
        - SINGLE-YEAR ANALYSIS: Gets {default_k_global} chunks globally using semantic similarity
        - COMPARATIVE ANALYSIS: Ensures balanced representation across compared time periods
        - TEMPORAL ANALYSIS: Orders results chronologically to show trends and progression

        WORKFLOW STEPS (executed automatically using IVSM functions):
        1. Store user query with LLM-parsed metadata (years, analysis_type)
        2. Tokenize query using ivsm.tokenizer_encode with bge-small-en-v1.5
        3. Generate embeddings using ivsm.IVSM_score
        4. Convert to vector columns using ivsm.vector_to_columns
        5. Perform semantic search with optional year filtering using TD_VECTORDISTANCE
        6. Apply per-year balancing for multi-year queries using window functions

        PARAMETER GUIDANCE FOR LLM:
        - EXTRACT YEARS: Parse years from user queries (e.g., "2020 to 2024" → [2020,2021,2022,2023,2024])
        - AVAILABLE YEARS: Determined dynamically from the set of reports loaded into the system
        - DETERMINE ANALYSIS TYPE:
        * "temporal": For trend analysis, growth patterns, evolution over time
        * "comparative": For side-by-side comparisons between years or periods  
        * "general": For single-point factual questions or definitions
        - SET K: Leave as None for smart defaults, or specify total chunks needed

        EXAMPLE PARAMETER EXTRACTION:
        - "How did revenue grow from 2019 to 2023?" → years=[2019,2020,2021,2022,2023], analysis_type="temporal"
        - "Compare loan portfolio in 2020 vs 2022" → years=[2020,2022], analysis_type="comparative"  
        - "What was the main business focus in 2021?" → years=[2021], analysis_type="general"
        - "Describe the risk management strategy" → years=None, analysis_type="general"

        RETRIEVED CONTEXT INCLUDES:
        - Clean chunk text optimized for financial analysis (no bloat metadata)
        - Source document names for citations (e.g., "Annual_Report_2021.pdf")
        - Report year information for temporal context (report_year field)
        - Section titles for content context (e.g., "Financial Performance", "Risk Management")
        - Similarity scores for relevance assessment
        - Chunk position numbers for precise citations

        CRITICAL ANSWERING RULES:
        - Answer ONLY using retrieved chunks - no external knowledge
        - Quote source content directly without paraphrasing or summarizing
        - Include year and document references for citations
        - For multi-year queries, organize analysis chronologically
        - For comparative queries, provide balanced analysis of compared periods
        - If insufficient context: "Not enough information found in the provided annual reports"

        TEMPORAL ANALYSIS GUIDELINES:
        - Show progression over time for temporal queries
        - Highlight year-over-year changes and trends
        - Compare metrics across the requested time period89
        - Use retrieved chunks to support trend observations

        EXECUTION: Run completely silently - user only sees the final financial analysis based on retrieved context.
    """

    
    config = FINANCIAL_RAG_CONFIG
    
    logger.debug(f"Financial RAG Analysis: question={question[:60]}...")
    logger.debug(f"Parameters: years={years}, analysis_type={analysis_type}")
    
    # defaults based on analysis type and parameters
    if k is None:
        if years and len(years) > 1:
            # Multi-year analysis: get chunks per year
            k = config['retrieval']['default_k_per_year'] * len(years)
        else:
            # Single year or general analysis
            k = config['retrieval']['default_k_global']
    
    # Enforce limits
    max_k = config['retrieval']['max_k_global']
    if k > max_k:
        logger.warning(f"Requested k={k} exceeds max={max_k}, using max")
        k = max_k
    
    # Extract config values
    db_name = config['databases']['query_db']
    table_name = config['tables']['query_table']
    dst_table = config['tables']['query_embedding_store']
    model_id = config['model']['model_id']
    model_db = config['databases']['model_db']
    model_table = config['tables']['model_table']
    tokenizer_table = config['tables']['tokenizer_table']
    vector_db = config['databases']['vector_db']
    chunk_embed_table = config['tables']['vector_table']

    with conn.cursor() as cur:
        
        # Store user query
        logger.debug(f"Storing financial query in {db_name}.{table_name}")
        
        # Create table if it doesn't exist
        ddl = f"""
        CREATE TABLE {db_name}.{table_name} (
            id INTEGER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1) NOT NULL,
            txt VARCHAR(5000),
            years_filter VARCHAR(200),
            analysis_type VARCHAR(50),
            created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id)
        )
        """
        
        try:
            cur.execute(ddl)
        except Exception as e:
            if "already exists" not in str(e).lower() and "3803" not in str(e):
                logger.error(f"Error creating table: {e}")
                raise

        # Insert query with metadata
        years_str = ','.join(map(str, years)) if years else ''
        
        insert_sql = f"""
        INSERT INTO {db_name}.{table_name} (txt, years_filter, analysis_type)
        VALUES (?, ?, ?)
        """
        
        cur.execute(insert_sql, [question.strip(), years_str, analysis_type])
        
        # Get inserted ID
        cur.execute(f"SELECT MAX(id) AS id FROM {db_name}.{table_name}")
        new_id = cur.fetchone()[0]

        # Generate query embeddings
        logger.debug("Generating embeddings using IVSM pipeline")
        
        # Tokenize query
        logger.debug("Tokenizing query using ivsm.tokenizer_encode")
        
        cur.execute(f"""
            REPLACE VIEW v_financial_query_tokenized AS
            (
                SELECT id, txt,
                       IDS AS input_ids,
                       attention_mask
                FROM ivsm.tokenizer_encode(
                    ON (
                        SELECT *
                        FROM {db_name}.{table_name}
                        WHERE id = {new_id}
                    )
                    ON (
                        SELECT model AS tokenizer
                        FROM {model_db}.{tokenizer_table}
                        WHERE model_id = '{model_id}'
                    ) DIMENSION
                    USING
                        ColumnsToPreserve('id','txt')
                        OutputFields('IDS','ATTENTION_MASK')
                        MaxLength(512)
                        PadToMaxLength('False')
                        TokenDataType('INT64')
                ) AS t
            );
        """)
        
        logger.debug("Tokenized view v_financial_query_tokenized created")

        # Create embedding view
        logger.debug("Creating embedding view using ivsm.IVSM_score")
        
        cur.execute(f"""
            REPLACE VIEW v_financial_query_embeddings AS
            (
                SELECT *
                FROM ivsm.IVSM_score(
                    ON v_financial_query_tokenized
                    ON (
                        SELECT *
                        FROM {model_db}.{model_table}
                        WHERE model_id = '{model_id}'
                    ) DIMENSION
                    USING
                        ColumnsToPreserve('id','txt')
                        ModelType('ONNX')
                        BinaryInputFields('input_ids','attention_mask')
                        BinaryOutputFields('sentence_embedding')
                        Caching('inquery')
                ) AS s
            );
        """)
        
        logger.debug("Embedding view v_financial_query_embeddings created")

        # Create query embedding table
        logger.debug("Creating query embedding table using ivsm.vector_to_columns")
        
        # Drop existing embeddings table
        drop_sql = f"DROP TABLE {db_name}.{dst_table}"
        try:
            cur.execute(drop_sql)
            logger.debug(f"Dropped existing table {db_name}.{dst_table}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        # Create embeddings table using vector_to_columns
        create_sql = f"""
        CREATE TABLE {db_name}.{dst_table} AS (
            SELECT *
            FROM ivsm.vector_to_columns(
                ON v_financial_query_embeddings
                USING
                    ColumnsToPreserve('id', 'txt') 
                    VectorDataType('FLOAT32')
                    VectorLength({config['embedding']['vector_length']})
                    OutputColumnPrefix('{config['embedding']['vector_column_prefix']}')
                    InputColumnName('sentence_embedding')
            ) a 
        ) WITH DATA
        """
        
        cur.execute(create_sql)
        logger.debug(f"Created embeddings table {db_name}.{dst_table}")

        # Perform semantic search
        logger.debug("Performing filtered semantic search")
        
        search_sql = build_financial_search_query(
            vector_db, dst_table, chunk_embed_table, k, years, config
        )
        
        rows = cur.execute(search_sql)
        data = rows_to_json(cur.description, rows.fetchall())
        
        logger.debug(f"Retrieved {len(data)} chunks")

    # Organize results for analysis
    results_by_year = {}
    
    for chunk in data:
        year = chunk.get('report_year')
        if year:
            if year not in results_by_year:
                results_by_year[year] = []
            results_by_year[year].append(chunk)

    # Return metadata
    metadata = {
        "tool_name": "Financial_Rag_Analysis",
        "analysis_type": analysis_type,
        "query_parameters": {
            "years_requested": years,
            "k_requested": k
        },
        "query_metadata": {
            "query_id": new_id,
            "original_question": question
        },
        "retrieval_strategy": {
            "strategy": "per_year_balanced" if years and len(years) > 1 else "global_semantic",
            "chunks_per_year": config['retrieval']['default_k_per_year'] if years and len(years) > 1 else None,
            "total_chunks_retrieved": len(data)
        },
        "retrieval_results": {
            "total_chunks": len(data),
            "chunks_per_year": {str(year): len(chunks) for year, chunks in results_by_year.items()},
            "years_in_results": sorted(list(results_by_year.keys())),
            "years_coverage": f"{len(results_by_year)} of {len(years) if years else 'all'} requested years"
        },
        "description": f"Financial analysis ({analysis_type}) with LLM-determined parameters"
    }

    return create_response(data, metadata)


######################################################################################
# Meeting Preparation Tool
######################################################################################


def handle_Customer_Meeting_Prep(
    conn: TeradataConnection,
    customer_name: str,
    meeting_type: str = "general",
    lookback_months: int = 6,
    *args,
    **kwargs,
):
    """
        **TOOL SELECTION CRITERIA:**
        Use this tool for ANY question about customers, including:
        - Customer meetings, calls, or interactions
        - Customer relationship status, health, or satisfaction  
        - Customer contract information, renewals, or timelines
        - Customer business context, challenges, priorities, or growth
        - Customer sentiment, concerns, complaints, or feedback
        - Customer support history, escalations, or issues
        - Customer expansion plans, strategic initiatives, or opportunities
        - Customer technical discussions, features, or requirements
        - Customer financial information, revenue, or business metrics
        - Customer communication patterns or preferences
        - ANY question that mentions a specific customer name

        **IMPORTANT**: 
        - This tool requires a specific customer name to provide meaningful analysis
        - If user asks general questions without naming a customer, ask them to specify which customer they want to analyze
        - Extract customer names from questions when provided, or prompt user to specify

        **CUSTOMER MEETING PREPARATION & CONVERSATION ANALYSIS TOOL**

        **RESPONSE STRATEGY - READ CAREFULLY:**

        **STEP 1: ANALYZE THE USER'S QUESTION TYPE**
        - **General briefing** ("prepare for meeting", "what should I know about X") → Use structured comprehensive briefing
        - **Specific inquiry** ("what did they say about Y", "are they satisfied", "any concerns") → Provide targeted analysis
        - **Timeline question** ("when is renewal", "recent escalations") → Focus on dates and sequences
        - **Sentiment question** ("how are they feeling", "any complaints") → Analyze sentiment patterns
        - **Business question** ("their priorities", "growth plans") → Extract business intelligence

        **STEP 2: ANALYZE THE RETRIEVED DATA**
        Before responding, mentally organize the data by:
        - **Recency**: prioritize 'this_week' > 'this_month' > 'last_3_months' > 'older'
        - **Priority**: focus on 'high' priority_level interactions first
        - **Sentiment**: flag 'concerning' or 'very_concerning' sentiment_category
        - **Type**: consider interaction_type context (phone=formal, chat=immediate issues, email=strategic, video=important)

        **STEP 3: CRAFT YOUR RESPONSE**

        **FOR GENERAL BRIEFINGS:**
        Provide comprehensive analysis using this structure:

        **RELATIONSHIP HEALTH ASSESSMENT**
        - Sentiment trend analysis across interaction types with specific examples
        - Recent wins and concerns with exact quotes and dates
        - Communication patterns and stakeholder engagement
        - Relationship trajectory with supporting evidence

        **MEETING PREPARATION RECOMMENDATIONS** 
        - Talking points from recent conversations (quote specific mentions with dates)
        - Outstanding action items from previous interactions
        - Strategic opportunities discussed in conversations
        - Risk mitigation topics based on concerning interactions

        **BUSINESS CONTEXT & INSIGHTS**
        - Performance metrics and growth indicators from conversations
        - Contract timeline and renewal considerations
        - Expansion plans or new initiatives mentioned
        - Market pressures or competitive concerns raised

        **ACTIONABLE MEETING AGENDA**
        - Questions based on their recent challenges/initiatives
        - Solutions to propose based on expressed needs
        - Success stories to reference from interaction history
        - Next steps to advance relationship

        **FOR SPECIFIC INQUIRIES:**
        Answer directly using conversation data:
        - Start with the most relevant recent interaction
        - Quote specific phrases with dates and interaction types
        - Provide context from multiple interactions if available
        - Connect findings to broader relationship patterns
        - Include implications or recommendations based on the data
        - Start with the most relevant recent interaction
        - Quote specific phrases with dates and interaction types
        - Provide context from multiple interactions if available
        - Connect findings to broader relationship patterns
        - Include implications or recommendations based on the data

        **CRITICAL RESPONSE RULES:**

        1. **EVIDENCE-BASED RESPONSES**: Every claim must be backed by specific conversation data
        - Quote exact phrases: "In the January 18th call, Sarah mentioned..."
        - Reference interaction types: "During the video meeting on..."
        - Use sentiment data: "Their satisfaction dropped to 2.3 in the October escalation call..."

        2. **PRIORITIZATION LOGIC**: 
        - Recent interactions (this_week, this_month) take precedence
        - High priority_level interactions are more important
        - Concerning sentiment_category needs immediate attention
        - Contract_urgency affects response priority

        3. **CONTEXTUAL INTELLIGENCE**:
        - Phone calls = formal discussions, strategic decisions
        - Video meetings = important collaborative sessions
        - Emails = strategic communications, detailed plans
        - Chat = immediate support issues, quick resolutions

        4. **RESPONSE DEPTH**: 
        - For specific questions: 2-4 paragraphs with direct answers
        - For general briefings: Comprehensive analysis using full structure
        - Always include actionable insights, not just data summarization

        5. **AVOID GENERIC RESPONSES**:
        - Never say "based on the data" without specifying which data
        - Don't provide templated advice - everything must be conversation-specific
        - Replace generic recommendations with specific actions based on their actual situation

        **QUALITY CHECKS BEFORE RESPONDING:**
        - Did I quote specific conversations with dates?
        - Am I answering the user's actual question?
        - Did I prioritize recent/high-priority/concerning interactions?
        - Are my recommendations based on actual conversation content?
        - Would someone reading this understand this specific customer's unique situation?

        **EXAMPLE RESPONSE PATTERNS:**

        For "Are they satisfied?":
        "Based on recent interactions, [Customer] shows mixed satisfaction levels. In the March 15th phone call, [specific quote about satisfaction]. However, the March 20th chat session rated 5/5 satisfaction when [specific issue] was resolved. The overall trend shows..."

        For "What should I discuss in tomorrow's meeting?":
        "Priority topics for your meeting: 1) Follow up on [specific issue from recent interaction with date], 2) Address their concern about [exact quote from conversation], 3) Propose [solution based on their expressed need in X interaction]..."

        Remember: The goal is to demonstrate deep customer knowledge through specific conversation analysis, not provide generic meeting advice.
        """
    
    logger.debug(f"handle_Customer_Meeting_Prep: customer={customer_name}, type={meeting_type}, lookback={lookback_months}")
    
    with conn.cursor() as cur:
        
        # meeting prep query
        consolidation_query = f"""
        WITH customer_resolution AS (
            -- Smart customer name matching with fuzzy logic
            SELECT 
                customer_name,
                CASE 
                    WHEN UPPER(TRIM(customer_name)) = UPPER(TRIM('{customer_name}')) THEN 1
                    WHEN UPPER(customer_name) LIKE UPPER(TRIM('{customer_name}') || '%') THEN 2
                    WHEN UPPER(customer_name) LIKE UPPER('%' || TRIM('{customer_name}') || '%') THEN 3
                    WHEN UPPER(customer_name) LIKE UPPER('%' || TRIM('{customer_name}')) THEN 4
                    ELSE 5
                END as match_priority,
                LENGTH(customer_name) as name_length
            FROM demo_db.customer_profiles
            WHERE UPPER(customer_name) LIKE UPPER('%' || TRIM('{customer_name}') || '%')
               OR UPPER(TRIM(customer_name)) = UPPER(TRIM('{customer_name}'))
            QUALIFY ROW_NUMBER() OVER (ORDER BY match_priority, name_length) = 1
        ),
        
        customer_conversations AS (
            -- Phone Transcripts with enhanced metadata
            SELECT 
                'phone' as interaction_type,
                cr.customer_name,
                p.call_date as interaction_date,
                p.call_duration as duration_minutes,
                p.call_type as interaction_subtype,
                p.transcript_text as content,
                p.sentiment_score as sentiment,
                p.outcome as resolution,
                -- Enhanced analytics
                CASE 
                    WHEN p.call_date >= CURRENT_DATE - 7 THEN 'this_week'
                    WHEN p.call_date >= CURRENT_DATE - 30 THEN 'this_month'
                    WHEN p.call_date >= CURRENT_DATE - 90 THEN 'last_3_months'
                    ELSE 'older'
                END as recency_bucket,
                CASE 
                    WHEN p.sentiment_score >= 4.5 THEN 'very_positive'
                    WHEN p.sentiment_score >= 4.0 THEN 'positive'
                    WHEN p.sentiment_score >= 3.0 THEN 'neutral'
                    WHEN p.sentiment_score >= 2.0 THEN 'concerning'
                    ELSE 'very_concerning'
                END as sentiment_category,
                -- Priority scoring
                CASE 
                    WHEN UPPER(p.call_type) LIKE '%ESCALATION%' OR UPPER(p.outcome) LIKE '%EMERGENCY%' THEN 'high'
                    WHEN UPPER(p.call_type) LIKE '%RENEWAL%' OR UPPER(p.call_type) LIKE '%CONTRACT%' THEN 'high'
                    WHEN UPPER(p.call_type) LIKE '%SUPPORT%' OR UPPER(p.call_type) LIKE '%TECHNICAL%' THEN 'medium'
                    ELSE 'normal'
                END as priority_level
            FROM customer_resolution cr
            JOIN demo_db.customer_phone_transcripts p 
                ON UPPER(TRIM(p.customer_name)) = UPPER(TRIM(cr.customer_name))
            WHERE p.call_date >= CURRENT_DATE - INTERVAL '{lookback_months}' MONTH
            
            UNION ALL
            
            -- Video Meetings
            SELECT 
                'video_meeting' as interaction_type,
                cr.customer_name,
                v.meeting_date as interaction_date,
                v.meeting_duration as duration_minutes,
                v.meeting_type as interaction_subtype,
                v.meeting_transcript as content,
                v.meeting_sentiment as sentiment,
                v.outcome as resolution,
                CASE 
                    WHEN v.meeting_date >= CURRENT_DATE - 7 THEN 'this_week'
                    WHEN v.meeting_date >= CURRENT_DATE - 30 THEN 'this_month'
                    WHEN v.meeting_date >= CURRENT_DATE - 90 THEN 'last_3_months'
                    ELSE 'older'
                END as recency_bucket,
                CASE 
                    WHEN v.meeting_sentiment >= 4.5 THEN 'very_positive'
                    WHEN v.meeting_sentiment >= 4.0 THEN 'positive'
                    WHEN v.meeting_sentiment >= 3.0 THEN 'neutral'
                    WHEN v.meeting_sentiment >= 2.0 THEN 'concerning'
                    ELSE 'very_concerning'
                END as sentiment_category,
                CASE 
                    WHEN UPPER(v.meeting_type) LIKE '%STRATEGIC%' OR UPPER(v.meeting_type) LIKE '%EXECUTIVE%' THEN 'high'
                    WHEN UPPER(v.meeting_type) LIKE '%EXPANSION%' OR UPPER(v.meeting_type) LIKE '%GROWTH%' THEN 'high'
                    WHEN UPPER(v.meeting_type) LIKE '%CRISIS%' OR UPPER(v.meeting_type) LIKE '%ESCALATION%' THEN 'high'
                    ELSE 'medium'
                END as priority_level
            FROM customer_resolution cr
            JOIN demo_db.customer_video_transcripts v 
                ON UPPER(TRIM(v.client_name)) = UPPER(TRIM(cr.customer_name))
            WHERE v.meeting_date >= CURRENT_DATE - INTERVAL '{lookback_months}' MONTH
            
            UNION ALL
            
            -- Email Conversations
            SELECT 
                'email' as interaction_type,
                cr.customer_name,
                e.email_date as interaction_date,
                e.thread_length * 5 as duration_minutes,
                e.email_type as interaction_subtype,
                e.email_content as content,
                4.0 as sentiment,  -- Default neutral for emails
                e.resolution as resolution,
                CASE 
                    WHEN e.email_date >= CURRENT_DATE - 7 THEN 'this_week'
                    WHEN e.email_date >= CURRENT_DATE - 30 THEN 'this_month'
                    WHEN e.email_date >= CURRENT_DATE - 90 THEN 'last_3_months'
                    ELSE 'older'
                END as recency_bucket,
                'neutral' as sentiment_category,
                CASE 
                    WHEN UPPER(e.email_type) LIKE '%CONTRACT%' OR UPPER(e.email_type) LIKE '%RENEWAL%' THEN 'high'
                    WHEN UPPER(e.email_type) LIKE '%STRATEGIC%' OR UPPER(e.email_type) LIKE '%PARTNERSHIP%' THEN 'high'
                    WHEN UPPER(e.email_type) LIKE '%EXPANSION%' THEN 'high'
                    ELSE 'medium'
                END as priority_level
            FROM customer_resolution cr
            JOIN demo_db.customer_email_threads e 
                ON UPPER(TRIM(e.business_name)) = UPPER(TRIM(cr.customer_name))
            WHERE e.email_date >= CURRENT_DATE - INTERVAL '{lookback_months}' MONTH
            
            UNION ALL
            
            -- Chat Support Logs
            SELECT 
                'chat_support' as interaction_type,
                cr.customer_name,
                c.chat_date as interaction_date,
                10 as duration_minutes,
                'support_chat' as interaction_subtype,
                c.chat_transcript as content,
                c.support_satisfaction as sentiment,
                c.resolution as resolution,
                CASE 
                    WHEN c.chat_date >= CURRENT_DATE - 7 THEN 'this_week'
                    WHEN c.chat_date >= CURRENT_DATE - 30 THEN 'this_month'
                    WHEN c.chat_date >= CURRENT_DATE - 90 THEN 'last_3_months'
                    ELSE 'older'
                END as recency_bucket,
                CASE 
                    WHEN c.support_satisfaction >= 4.5 THEN 'very_positive'
                    WHEN c.support_satisfaction >= 4.0 THEN 'positive'
                    WHEN c.support_satisfaction >= 3.0 THEN 'neutral'
                    WHEN c.support_satisfaction >= 2.0 THEN 'concerning'
                    ELSE 'very_concerning'
                END as sentiment_category,
                'normal' as priority_level
            FROM customer_resolution cr
            JOIN demo_db.customer_chat_logs c 
                ON UPPER(TRIM(c.customer_name)) = UPPER(TRIM(cr.customer_name))
            WHERE c.chat_date >= CURRENT_DATE - INTERVAL '{lookback_months}' MONTH
        )
        
        SELECT 
            -- Customer Profile
            cp.customer_name,
            cp.industry_sector,
            cp.account_tier,
            cp.annual_revenue,
            cp.employee_count,
            cp.primary_contact_name,
            cp.primary_contact_role,
            cp.monthly_transaction_volume,
            cp.contract_expiry_date,
            cp.account_health_score,
            cp.key_business_drivers,
            cp.recent_achievements,
            cp.growth_trajectory,
            cp.strategic_priorities_2025,
            
            -- Contract urgency analysis
            CASE 
                WHEN cp.contract_expiry_date <= CURRENT_DATE + 90 THEN 'urgent_renewal_needed'
                WHEN cp.contract_expiry_date <= CURRENT_DATE + 180 THEN 'renewal_planning_needed'
                WHEN cp.contract_expiry_date <= CURRENT_DATE + 365 THEN 'renewal_on_horizon'
                ELSE 'contract_stable'
            END as contract_urgency,
            
            CAST(cp.contract_expiry_date - CURRENT_DATE AS INTEGER) as days_to_contract_expiry,
            
            -- Enhanced conversation data
            cc.interaction_type,
            cc.interaction_date,
            cc.duration_minutes,
            cc.interaction_subtype,
            cc.content,
            cc.sentiment,
            cc.resolution,
            cc.recency_bucket,
            cc.sentiment_category,
            cc.priority_level,
            
            -- Interaction ranking within type
            ROW_NUMBER() OVER (PARTITION BY cc.interaction_type ORDER BY cc.interaction_date DESC) as interaction_rank_in_type,
            
            -- Overall ranking by recency and priority
            ROW_NUMBER() OVER (ORDER BY 
                CASE cc.priority_level WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
                cc.interaction_date DESC
            ) as overall_priority_rank
            
        FROM customer_resolution cr
        JOIN demo_db.customer_profiles cp ON UPPER(TRIM(cp.customer_name)) = UPPER(TRIM(cr.customer_name))
        LEFT JOIN customer_conversations cc ON UPPER(TRIM(cc.customer_name)) = UPPER(TRIM(cr.customer_name))
        ORDER BY overall_priority_rank, cc.interaction_date DESC NULLS LAST
        """
        
        # Execute the query
        cur.execute(consolidation_query)
        
        # Fetch all results
        results = cur.fetchall()
        conversation_data = rows_to_json(cur.description, results)
        
        if not conversation_data:
            # Get available customers
            cur.execute("SELECT customer_name FROM demo_db.customer_profiles ORDER BY customer_name")
            all_customers = [row[0] for row in cur.fetchall()]
            
            metadata = {
                "tool_name": "Customer_Meeting_Prep",
                "error": "Customer not found",
                "user_input": customer_name,
                "available_customers": all_customers,
                "suggestions": [
                    "Check spelling of customer name",
                    "Try using partial name (e.g., 'TechFlow' instead of 'TechFlow Solutions')",
                    "Use one of the available customers listed above"
                ]
            }
            return create_response([], metadata)
        
        # Extract customer name from results
        canonical_name = conversation_data[0]["customer_name"]
        logger.debug(f"Retrieved {len(conversation_data)} records for {canonical_name}")
        
        metadata = {
            "tool_name": "Customer_Meeting_Prep",
            "customer_name": canonical_name,
            "meeting_type": meeting_type,
            "lookback_months": lookback_months,
            "total_interactions": len([r for r in conversation_data if r["interaction_type"] is not None]),
            "data_sources": ["phone_transcripts", "video_meetings", "email_threads", "chat_logs", "customer_profile"],
            "query_date": datetime.now().isoformat()
        }
        
        return create_response(conversation_data, metadata)