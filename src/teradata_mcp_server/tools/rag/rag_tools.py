import logging
import yaml
import os
from typing import Optional, Any, Dict, List
from teradatasql import TeradataConnection
import json
from datetime import date, datetime
from decimal import Decimal


logger = logging.getLogger("teradata_mcp_server")

# Load RAG configuration
def load_rag_config():
    """Load RAG configuration from rag_config.yaml"""
    try:
        with open('rag_config.yaml', 'r') as file:  # Simple path like server.py
            logger.info("Loading RAG config from: rag_config.yaml")
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning("RAG config file not found: rag_config.yaml, using defaults")
        return get_default_rag_config()
    except Exception as e:
        logger.error(f"Error loading RAG config: {e}")
        return get_default_rag_config()

def get_default_rag_config():
    """Default RAG configuration as fallback"""
    return {
        'version': 'byom',
        'databases': {
            'query_db': 'demo_db',
            'model_db': 'demo_db', 
            'vector_db': 'demo_db'
        },
        'tables': {
            'query_table': 'user_query',
            'query_embedding_store': 'user_query_embeddings',
            'vector_table': 'icici_fr_embeddings_store',
            'model_table': 'embeddings_models',
            'tokenizer_table': 'embeddings_tokenizers'
        },
        'model': {
            'model_id': 'bge-small-en-v1.5'
        },
        'retrieval': {
            'default_k': 10,
            'max_k': 50
        },
        'vector_store_schema': {
            'required_fields': ['txt'],
            'metadata_fields_in_vector_store': ['chunk_num', 'section_title', 'doc_name']
        },
        'embedding': {
            'vector_length': 384,
            'vector_column_prefix': 'emb_',
            'distance_measure': 'cosine',
            'feature_columns': '[emb_0:emb_383]'
        }
    }

# Load config at module level
RAG_CONFIG = load_rag_config()

def build_search_query(vector_db, dst_table, chunk_embed_table, k, config):
    """Build dynamic search query based on available metadata fields in vector store"""
    # Get metadata fields from config
    metadata_fields = config['vector_store_schema']['metadata_fields_in_vector_store']
    feature_columns = config['embedding']['feature_columns']
    
    # Build SELECT clause dynamically - txt is always required
    select_fields = ["e_ref.txt AS reference_txt"]
    
    # Add all metadata fields from vector store
    for field in metadata_fields:
        # Skip txt since it's already added as reference_txt
        if field != 'txt':
            select_fields.append(f"e_ref.{field} AS {field}")
    
    # Always add similarity (calculated field)
    select_fields.append("(1.0 - dt.distance) AS similarity")
    
    select_clause = ",\n            ".join(select_fields)
    
    return f"""
        SELECT
            {select_clause}
        FROM TD_VECTORDISTANCE (
                ON {vector_db}.{dst_table}      AS TargetTable
                ON {vector_db}.{chunk_embed_table}      AS ReferenceTable DIMENSION
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
        ORDER BY similarity DESC;
        """

def serialize_teradata_types(obj: Any) -> Any:
    """Convert Teradata-specific types to JSON serializable formats"""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)

def rows_to_json(cursor_description: Any, rows: List[Any]) -> List[Dict[str, Any]]:
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

def create_response(data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
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

def handle_rag_executeWorkflow(
    conn: TeradataConnection,
    question: str,
    k: int = None,
    *args,
    **kwargs,
):
    """
    Execute complete RAG workflow to answer user questions based on document context.

    This function handles the entire RAG pipeline:
    1. Configuration setup (using configurable values from rag_config.yaml)
    2. Store user query (with /rag prefix stripping)
    3. Generate query embeddings (tokenization + embedding)
    4. Perform semantic search against chunk embeddings
    5. Return retrieved context chunks for answer generation

    The function uses configuration values from rag_config.yaml with fallback defaults.

    Returns the top-k most relevant chunks with metadata for context-grounded answer generation.
    """
    
    # Use configuration from loaded config
    config = RAG_CONFIG
    
    # Use config default if k not provided
    if k is None:
        k = config['retrieval']['default_k']
    
    # Optional: Enforce max limit
    max_k = config['retrieval'].get('max_k', 100)
    if k > max_k:
        logger.warning(f"Requested k={k} exceeds max_k={max_k}, using max_k")
        k = max_k
    
    logger.debug(f"handle_rag_executeWorkflow: question={question[:60]}..., k={k}")
    
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
        
        # Step 2: Store user query
        logger.debug(f"Step 2: Storing user query in {db_name}.{table_name}")
        
        # Create table if it doesn't exist
        ddl = f"""
        CREATE TABLE {db_name}.{table_name} (
            id INTEGER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1) NOT NULL,
            txt VARCHAR(5000),
            created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id)
        )
        """
        
        try:
            cur.execute(ddl)
            logger.debug(f"Table {db_name}.{table_name} created")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "3803" in error_msg:
                logger.debug(f"Table {db_name}.{table_name} already exists, skipping creation")
            else:
                logger.error(f"Error creating table: {e}")
                raise

        # Insert cleaned question
        insert_sql = f"""
        INSERT INTO {db_name}.{table_name} (txt)
        SELECT
          CASE
            WHEN TRIM(?) LIKE '/rag %' THEN SUBSTRING(TRIM(?) FROM 6)
            ELSE TRIM(?)
          END
        """
        cur.execute(insert_sql, [question, question, question])
        
        # Get inserted ID and cleaned text
        cur.execute(f"SELECT MAX(id) AS id FROM {db_name}.{table_name}")
        new_id = cur.fetchone()[0]
        
        cur.execute(f"SELECT txt FROM {db_name}.{table_name} WHERE id = ?", [new_id])
        cleaned_txt = cur.fetchone()[0]
        
        logger.debug(f"Stored query with ID {new_id}: {cleaned_txt[:60]}...")

        # Step 3: Generate query embeddings
        logger.debug(f"Step 3: Generating embeddings in {db_name}.{dst_table}")
        
        # Drop existing embeddings table
        drop_sql = f"DROP TABLE {db_name}.{dst_table}"
        try:
            cur.execute(drop_sql)
            logger.debug(f"Dropped existing table {db_name}.{dst_table}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        # Create embeddings table using ONNXEmbeddings
        create_sql = f"""
        CREATE TABLE {db_name}.{dst_table} AS (
            SELECT *
            FROM mldb.ONNXEmbeddings(
                ON (SELECT id, txt FROM {db_name}.{table_name})
                ON (SELECT model_id, model FROM {model_db}.{model_table} WHERE model_id = '{model_id}') DIMENSION
                ON (SELECT model AS tokenizer FROM {model_db}.{tokenizer_table} WHERE model_id = '{model_id}') DIMENSION
                USING
                    Accumulate('id', 'txt')
                    ModelOutputTensor('sentence_embedding')
                    OutputFormat('FLOAT32({config["embedding"]["vector_length"]})')
            ) AS a
        ) WITH DATA
        """
        
        cur.execute(create_sql)
        logger.debug(f"Created embeddings table {db_name}.{dst_table}")

        # Step 4: Perform semantic search with dynamic query building
        logger.debug(f"Step 4: Performing semantic search with k={k}")
        
        search_sql = build_search_query(vector_db, dst_table, chunk_embed_table, k, config)
        
        rows = cur.execute(search_sql)
        data = rows_to_json(cur.description, rows.fetchall())
        
        logger.debug(f"Retrieved {len(data)} chunks for semantic search")

    # Return results with comprehensive metadata
    metadata = {
        "tool_name": "rag_executeWorkflow",
        "workflow_steps": ["config_set", "query_stored", "embeddings_generated", "semantic_search_completed"],
        "query_id": new_id,
        "cleaned_question": cleaned_txt,
        "database": db_name,
        "query_table": table_name,
        "embedding_table": dst_table,
        "vector_table": chunk_embed_table,
        "model_id": model_id,
        "chunks_retrieved": len(data),
        "topk_requested": k,
        "topk_configured_default": config['retrieval']['default_k'],
        "metadata_fields": config['vector_store_schema']['metadata_fields_in_vector_store'],
        "description": "Complete RAG workflow executed: config → store query → generate embeddings → semantic search"
    }

    return create_response(data, metadata)

def handle_rag_executeWorkflow_ivsm(
    conn: TeradataConnection,
    question: str,
    k: int = None,
    *args,
    **kwargs,
):
    """
    Execute complete RAG workflow to answer user questions based on document context.

    This function handles the entire RAG pipeline using IVSM functions:
    1. Configuration setup (using configurable values from rag_config.yaml)
    2. Store user query (with /rag prefix stripping)
    3. Tokenize query using ivsm.tokenizer_encode
    4. Create embedding view using ivsm.IVSM_score
    5. Convert embeddings to vector columns using ivsm.vector_to_columns
    6. Perform semantic search against chunk embeddings

    The function uses configuration values from rag_config.yaml with fallback defaults.

    Returns the top-k most relevant chunks with metadata for context-grounded answer generation.
    """
    
    # Use configuration from loaded config
    config = RAG_CONFIG
    
    # Use config default if k not provided
    if k is None:
        k = config['retrieval']['default_k']
    
    # Optional: Enforce max limit
    max_k = config['retrieval'].get('max_k', 100)
    if k > max_k:
        logger.warning(f"Requested k={k} exceeds max_k={max_k}, using max_k")
        k = max_k
    
    logger.debug(f"handle_rag_executeWorkflow (IVSM): question={question[:60]}..., k={k}")
    
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
        
        # Step 2: Store user query
        logger.debug(f"Step 2: Storing user query in {db_name}.{table_name}")
        
        # Create table if it doesn't exist
        ddl = f"""
        CREATE TABLE {db_name}.{table_name} (
            id INTEGER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1) NOT NULL,
            txt VARCHAR(5000),
            created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id)
        )
        """
        
        try:
            cur.execute(ddl)
            logger.debug(f"Table {db_name}.{table_name} created")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "3803" in error_msg:
                logger.debug(f"Table {db_name}.{table_name} already exists, skipping creation")
            else:
                logger.error(f"Error creating table: {e}")
                raise

        # Insert cleaned question
        insert_sql = f"""
        INSERT INTO {db_name}.{table_name} (txt)
        SELECT
          CASE
            WHEN TRIM(?) LIKE '/rag %' THEN SUBSTRING(TRIM(?) FROM 6)
            ELSE TRIM(?)
          END
        """
        cur.execute(insert_sql, [question, question, question])
        
        # Get inserted ID and cleaned text
        cur.execute(f"SELECT MAX(id) AS id FROM {db_name}.{table_name}")
        new_id = cur.fetchone()[0]
        
        cur.execute(f"SELECT txt FROM {db_name}.{table_name} WHERE id = ?", [new_id])
        cleaned_txt = cur.fetchone()[0]
        
        logger.debug(f"Stored query with ID {new_id}: {cleaned_txt[:60]}...")

        # Step 3: Tokenize query
        logger.debug(f"Step 3: Tokenizing query using ivsm.tokenizer_encode")
        
        cur.execute(f"""
            REPLACE VIEW v_topics_tokenized AS
            (
                SELECT id, txt,
                       IDS AS input_ids,
                       attention_mask
                FROM ivsm.tokenizer_encode(
                    ON (
                        SELECT *
                        FROM {db_name}.{table_name}
                        QUALIFY ROW_NUMBER() OVER (ORDER BY created_ts DESC) = 1
                    )
                    ON (
                        SELECT model AS tokenizer
                        FROM {model_db}.{tokenizer_table}
                        WHERE model_id = '{model_id}'
                    ) DIMENSION
                    USING
                        ColumnsToPreserve('id','txt')
                        OutputFields('IDS','ATTENTION_MASK')
                        MaxLength(1024)
                        PadToMaxLength('True')
                        TokenDataType('INT64')
                ) AS t
            );
        """)
        
        logger.debug("Tokenized view v_topics_tokenized created")

        # Step 4: Create embedding view
        logger.debug(f"Step 4: Creating embedding view using ivsm.IVSM_score")
        
        cur.execute(f"""
            REPLACE VIEW v_topics_embeddings AS
            (
                SELECT *
                FROM ivsm.IVSM_score(
                    ON v_topics_tokenized
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
        
        logger.debug("Embedding view v_topics_embeddings created")

        # Step 5: Create query embedding table
        logger.debug(f"Step 5: Creating query embedding table using ivsm.vector_to_columns")
        
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
                ON v_topics_embeddings
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

        # Step 6: Perform semantic search with dynamic query building
        logger.debug(f"Step 6: Performing semantic search with k={k}")
        
        search_sql = build_search_query(vector_db, dst_table, chunk_embed_table, k, config)
        
        rows = cur.execute(search_sql)
        data = rows_to_json(cur.description, rows.fetchall())
        
        logger.debug(f"Retrieved {len(data)} chunks for semantic search")

    # Return results with comprehensive metadata
    metadata = {
        "tool_name": "rag_executeWorkflow_ivsm",
        "workflow_type": "IVSM",
        "workflow_steps": ["config_set", "query_stored", "query_tokenized", "embedding_view_created", "embedding_table_created", "semantic_search_completed"],
        "query_id": new_id,
        "cleaned_question": cleaned_txt,
        "database": db_name,
        "query_table": table_name,
        "embedding_table": dst_table,
        "vector_table": chunk_embed_table,
        "model_id": model_id,
        "chunks_retrieved": len(data),
        "topk_requested": k,
        "topk_configured_default": config['retrieval']['default_k'],
        "views_created": ["v_topics_tokenized", "v_topics_embeddings"],
        "metadata_fields": config['vector_store_schema']['metadata_fields_in_vector_store'],
        "description": "Complete RAG workflow executed using IVSM functions: config → store query → tokenize → create embedding view → create embedding table → semantic search"
    }

    return create_response(data, metadata)

def handle_customer_meetingPrep(
    conn: TeradataConnection,
    customer_name: str,
    meeting_type: str = "general",
    lookback_months: int = 6,
    *args,
    **kwargs,
):
    """
    Consolidate all customer conversation data for AI-powered meeting preparation recommendations.
    
    This function executes a comprehensive SQL query to gather ALL conversation data for a specific customer
    across phone calls, video meetings, emails, and chat logs, with enhanced analytics for AI analysis.
    
    Returns structured data with customer profile, conversation history, and analytical insights
    for the LLM to generate specific, actionable meeting preparation recommendations.
    """
    
    logger.debug(f"handle_customer_meetingPrep: customer={customer_name}, type={meeting_type}, lookback={lookback_months}")
    
    with conn.cursor() as cur:
        
        # Enhanced customer meeting prep query with smart name resolution and analytics
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
        
        # Execute the consolidation query
        cur.execute(consolidation_query)
        
        # Fetch all results
        results = cur.fetchall()
        conversation_data = rows_to_json(cur.description, results)
        
        if not conversation_data:
            # Get available customers for helpful error message
            cur.execute("SELECT customer_name FROM demo_db.customer_profiles ORDER BY customer_name")
            all_customers = [row[0] for row in cur.fetchall()]
            
            metadata = {
                "tool_name": "customer_meetingPrep",
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
            "tool_name": "customer_meetingPrep",
            "customer_name": canonical_name,
            "meeting_type": meeting_type,
            "lookback_months": lookback_months,
            "total_interactions": len([r for r in conversation_data if r["interaction_type"] is not None]),
            "data_sources": ["phone_transcripts", "video_meetings", "email_threads", "chat_logs", "customer_profile"],
            "query_date": datetime.now().isoformat()
        }
        
        return create_response(conversation_data, metadata)
    


##################################################################################
# SQL Clustering Pipeline
##################################################################################

import logging
import yaml
from typing import Optional, Any, Dict, List

import json
from datetime import date, datetime
from decimal import Decimal

logger = logging.getLogger("teradata_mcp_server")

# Load SQL Clustering configuration
def load_sql_clustering_config():
    """Load SQL clustering configuration from sql_clustering_config.yaml"""
    try:
        with open('sql_clustering_config.yaml', 'r') as file:
            logger.info("Loading SQL clustering config from: sql_clustering_config.yaml")
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning("SQL clustering config file not found: sql_clustering_config.yaml, using defaults")
        return get_default_sql_clustering_config()
    except Exception as e:
        logger.error(f"Error loading SQL clustering config: {e}")
        return get_default_sql_clustering_config()

def get_default_sql_clustering_config():
    """Default SQL clustering configuration as fallback"""
    return {
        'version': 'ivsm',
        'databases': {
            'feature_db': 'feature_ext_db',
            'model_db': 'feature_ext_db'
        },
        'tables': {
            'sql_query_log_main': 'sql_query_log_main',
            'sql_log_tokenized_for_embeddings': 'sql_log_tokenized_for_embeddings',
            'sql_log_embeddings': 'sql_log_embeddings',
            'sql_log_embeddings_store': 'sql_log_embeddings_store',
            'sql_query_clusters_temp': 'sql_query_clusters_temp',
            'sql_query_clusters': 'sql_query_clusters',
            'query_cluster_stats': 'query_cluster_stats',
            'embedding_models': 'embedding_models',
            'embedding_tokenizers': 'embedding_tokenizers'
        },
        'model': {
            'model_id': 'bge-small-en-v1.5'
        },
        'clustering': {
            'optimal_k': 14,
            'max_queries': 10000,
            'seed': 10,
            'stop_threshold': 0.0395,
            'max_iterations': 100
        },
        'embedding': {
            'vector_length': 384,
            'max_length': 1024,
            'pad_to_max_length': 'False'
        }
    }

# Load config at module level
SQL_CLUSTERING_CONFIG = load_sql_clustering_config()

def handle_sql_clustering_executeFullPipeline(
    conn,
    optimal_k: int = None,
    max_queries: int = None,
    *args,
    **kwargs
):
    """
    Execute the complete SQL query clustering pipeline from query log extraction through cluster statistics generation.
    
    This function performs the entire workflow:
    1. Extract SQL query log with performance metrics
    2. Tokenize queries for embeddings
    3. Generate embeddings using IVSM functions
    4. Convert embeddings to vector store format
    5. Perform K-means clustering with optimal K
    6. Calculate silhouette scores
    7. Generate cluster statistics
    
    Returns comprehensive cluster analysis for high CPU usage query optimization.
    """
    
    config = SQL_CLUSTERING_CONFIG
    
    # Use config defaults if not provided
    if optimal_k is None:
        optimal_k = config['clustering']['optimal_k']
    if max_queries is None:
        max_queries = config['clustering']['max_queries']
    
    logger.debug(f"handle_sql_clustering_executeFullPipeline: optimal_k={optimal_k}, max_queries={max_queries}")
    
    # Extract config values
    feature_db = config['databases']['feature_db']
    model_db = config['databases']['model_db']
    model_id = config['model']['model_id']
    
    tables = config['tables']
    embedding_config = config['embedding']
    clustering_config = config['clustering']

    with conn.cursor() as cur:
        
        # Step 1: Create main SQL query log table (OPTIMIZED)
        logger.debug(f"Step 1: Creating main query log table {feature_db}.{tables['sql_query_log_main']}")
        
        try:
            cur.execute(f"DROP TABLE {feature_db}.{tables['sql_query_log_main']}")
            logger.debug(f"Dropped existing table {feature_db}.{tables['sql_query_log_main']}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        main_query_sql = f"""
        CREATE TABLE {feature_db}.{tables['sql_query_log_main']} AS (
            SELECT 
                CAST(a.QueryID AS BIGINT) AS id,
                a.SQLTextInfo AS txt,
                b.username,
                b.appid,
                b.numsteps,
                b.ampcputime,
                b.TotalIOCount AS logicalio,
                b.wdname,
                CASE WHEN b.ampcputime < HashAmp()+1 OR (b.ampcputime / (HashAmp()+1)) = 0 
                     THEN 0 ELSE b.maxampcputime/(b.ampcputime / (HashAmp()+1)) END (DEC(8,2)) AS CPUSKW,
                CASE WHEN b.ampcputime < HashAmp()+1 OR (b.TotalIOCount / (HashAmp()+1)) = 0 
                     THEN 0 ELSE b.maxampio/(b.TotalIOCount / (HashAmp()+1)) END (DEC(8,2)) AS IOSKW,
                CASE WHEN b.ampcputime < HashAmp()+1 OR b.TotalIOCount = 0 
                     THEN 0 ELSE (b.ampcputime * 1000)/b.TotalIOCount END AS PJI,
                CASE WHEN b.ampcputime < HashAmp()+1 OR b.ampcputime = 0 
                     THEN 0 ELSE b.TotalIOCount/(b.ampcputime * 1000) END AS UII,
                CAST(EXTRACT(HOUR FROM ((b.FirstRespTime - b.StartTime) HOUR(3) TO SECOND(6))) * 3600
                     + EXTRACT(MINUTE FROM ((b.FirstRespTime - b.StartTime) HOUR(3) TO SECOND(6))) * 60
                     + EXTRACT(SECOND FROM ((b.FirstRespTime - b.StartTime) HOUR(3) TO SECOND(6))) AS DECIMAL(10,2)) AS response_secs,
                (CAST(EXTRACT(HOUR FROM ((b.FirstRespTime - b.StartTime) HOUR(3) TO SECOND(6))) * 3600
                     + EXTRACT(MINUTE FROM ((b.FirstRespTime - b.StartTime) HOUR(3) TO SECOND(6))) * 60
                     + EXTRACT(SECOND FROM ((b.FirstRespTime - b.StartTime) HOUR(3) TO SECOND(6))) AS DECIMAL(10,2)))/60.0 AS response_mins,
                CASE WHEN b.delaytime IS NULL THEN 0.0 ELSE b.delaytime END AS delaytime
            FROM DBC.DBQLSqlTbl a 
            JOIN (
                -- OPTIMIZATION: Filter to top queries by CPU BEFORE joining
                SELECT * FROM DBC.DBQLOgTbl 
                WHERE LOWER(statementtype) IN ('select','create table')
                QUALIFY ROW_NUMBER() OVER (ORDER BY ampcputime DESC) <= {max_queries}
            ) b ON a.queryid = b.queryid AND a.procid = b.procid
            WHERE
                a.SQLTextInfo NOT LIKE '%SET QUERY_BAND%' AND
                a.SQLTextInfo NOT LIKE '%ParamValue%' AND
                a.SQLTextInfo NOT LIKE '%SELECT CURRENT_TIMESTAMP%' AND
                LOWER(a.SQLTextInfo) NOT LIKE '%dbc.%' AND 
                a.SqlRowNo = 1
        ) WITH DATA
        """
        
        cur.execute(main_query_sql)
        logger.debug(f"Created main query log table")

        # Step 2: Create tokenized table for embeddings (SIMPLIFIED - no additional filtering needed)
        logger.debug(f"Step 2: Creating tokenized table {feature_db}.{tables['sql_log_tokenized_for_embeddings']}")
        
        try:
            cur.execute(f"DROP TABLE {feature_db}.{tables['sql_log_tokenized_for_embeddings']}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        tokenize_sql = f"""
        CREATE TABLE {feature_db}.{tables['sql_log_tokenized_for_embeddings']} AS (
            SELECT
                id,
                txt,
                IDS AS input_ids,
                attention_mask
            FROM ivsm.tokenizer_encode(
                ON (SELECT * FROM {feature_db}.{tables['sql_query_log_main']})
                ON (SELECT model AS tokenizer FROM {model_db}.{tables['embedding_tokenizers']} 
                    WHERE model_id = '{model_id}') DIMENSION
                USING
                    ColumnsToPreserve('id', 'txt')
                    OutputFields('IDS', 'ATTENTION_MASK')
                    MaxLength({embedding_config['max_length']})
                    PadToMaxLength('{embedding_config['pad_to_max_length']}')
                    TokenDataType('INT64')
            ) AS dt
        ) WITH DATA
        """
        
        cur.execute(tokenize_sql)
        logger.debug(f"Created tokenized table")

        # Step 3: Create embeddings table
        logger.debug(f"Step 3: Creating embeddings table {feature_db}.{tables['sql_log_embeddings']}")
        
        try:
            cur.execute(f"DROP TABLE {feature_db}.{tables['sql_log_embeddings']}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        embeddings_sql = f"""
        CREATE TABLE {feature_db}.{tables['sql_log_embeddings']} AS (
            SELECT *
            FROM ivsm.IVSM_score(
                ON {feature_db}.{tables['sql_log_tokenized_for_embeddings']}
                ON (SELECT * FROM {model_db}.{tables['embedding_models']} 
                    WHERE model_id = '{model_id}') DIMENSION
                USING
                    ColumnsToPreserve('id', 'txt')
                    ModelType('ONNX')
                    BinaryInputFields('input_ids', 'attention_mask')
                    BinaryOutputFields('sentence_embedding')
                    Caching('inquery')
            ) a
        ) WITH DATA
        """
        
        cur.execute(embeddings_sql)
        logger.debug(f"Created embeddings table")

        # Step 4: Create embeddings store table
        logger.debug(f"Step 4: Creating embeddings store table {feature_db}.{tables['sql_log_embeddings_store']}")
        
        try:
            cur.execute(f"DROP TABLE {feature_db}.{tables['sql_log_embeddings_store']}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        embeddings_store_sql = f"""
        CREATE TABLE {feature_db}.{tables['sql_log_embeddings_store']} AS (
            SELECT *
            FROM ivsm.vector_to_columns(
                ON {feature_db}.{tables['sql_log_embeddings']}
                USING
                    ColumnsToPreserve('id', 'txt')
                    VectorDataType('FLOAT32')
                    VectorLength({embedding_config['vector_length']})
                    OutputColumnPrefix('emb_')
                    InputColumnName('sentence_embedding')
            ) a
        ) WITH DATA
        """
        
        cur.execute(embeddings_store_sql)
        logger.debug(f"Created embeddings store table")

        # Step 5: Perform K-means clustering
        logger.debug(f"Step 5: Performing K-means clustering with k={optimal_k}")
        
        try:
            cur.execute(f"DROP TABLE {feature_db}.{tables['sql_query_clusters_temp']}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        kmeans_sql = f"""
        CREATE TABLE {feature_db}.{tables['sql_query_clusters_temp']} AS (
            SELECT td_clusterid_kmeans, a.*
            FROM TD_KMeans (
                ON {feature_db}.{tables['sql_log_embeddings_store']} AS InputTable
                USING
                    IdColumn('id')
                    TargetColumns('[2:385]')
                    NumClusters({optimal_k})
                    Seed({clustering_config['seed']})
                    StopThreshold({clustering_config['stop_threshold']})
                    OutputClusterAssignment('true')
                    MaxIterNum({clustering_config['max_iterations']})
            ) AS dt
            JOIN {feature_db}.{tables['sql_query_log_main']} a ON a.id = dt.id
        ) WITH DATA
        """
        
        cur.execute(kmeans_sql)
        logger.debug(f"Created temporary clusters table")

        # Step 6: Create final clusters table with silhouette scores
        logger.debug(f"Step 6: Creating final clusters table with silhouette scores")
        
        try:
            cur.execute(f"DROP TABLE {feature_db}.{tables['sql_query_clusters']}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        final_clusters_sql = f"""
        CREATE TABLE {feature_db}.{tables['sql_query_clusters']} AS (
            SELECT a.*, b.silhouette_score 
            FROM {feature_db}.{tables['sql_query_clusters_temp']} a
            JOIN (SELECT * FROM TD_Silhouette(
                ON (SELECT td_clusterid_kmeans, b.* 
                    FROM {feature_db}.{tables['sql_query_clusters_temp']} a 
                    JOIN {feature_db}.{tables['sql_log_embeddings_store']} b
                    ON a.id = b.id) AS InputTable
                USING
                    IdColumn('id')
                    ClusterIdColumn('td_clusterid_kmeans')
                    TargetColumns('[4:]')
                    OutputType('SAMPLE_SCORES')
            ) AS dt) AS b
            ON a.id = b.id
        ) WITH DATA PRIMARY INDEX(id)
        """
        
        cur.execute(final_clusters_sql)
        logger.debug(f"Created final clusters table")

        # Step 7: Create cluster statistics table
        logger.debug(f"Step 7: Creating cluster statistics table")
        
        try:
            cur.execute(f"DROP TABLE {feature_db}.{tables['query_cluster_stats']}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        cluster_stats_sql = f"""
        CREATE TABLE {feature_db}.{tables['query_cluster_stats']} AS (
            SELECT a.td_clusterid_kmeans,
                AVG(a.numsteps) AS avg_numsteps, 
                VAR_SAMP(a.numsteps) AS var_numsteps,
                AVG(a.ampcputime) AS avg_cpu, 
                VAR_SAMP(a.ampcputime) AS var_cpu,
                AVG(a.logicalio) AS avg_io, 
                VAR_SAMP(a.logicalio) AS var_io,
                AVG(a.cpuskw) AS avg_cpuskw, 
                VAR_SAMP(a.cpuskw) AS var_cpuskw,
                AVG(a.ioskw) AS avg_ioskw, 
                VAR_SAMP(a.ioskw) AS var_ioskw,
                AVG(a.pji) AS avg_pji, 
                VAR_SAMP(a.pji) AS var_pji,
                AVG(a.uii) AS avg_uii, 
                VAR_SAMP(a.uii) AS var_uii,
                MAX(un.top_username) AS top_username,
                MAX(top_wdname) AS top_wdname,
                MAX(top_appid) AS top_appid,
                MAX(s1.silhouette_score) AS overall_silhouette_score,
                MAX(s2.silhouette_score) AS cluster_silhouette_score,
                COUNT(*) AS queries
            FROM {feature_db}.{tables['sql_query_clusters']} a 
            JOIN (
                SELECT td_clusterid_kmeans, 
                       username AS top_UserName
                FROM {feature_db}.{tables['sql_query_clusters']}
                GROUP BY td_clusterid_kmeans, username
                QUALIFY ROW_NUMBER() OVER (PARTITION BY td_clusterid_kmeans ORDER BY COUNT(*) DESC) = 1
            ) un ON a.td_clusterid_kmeans = un.td_clusterid_kmeans
            JOIN (
                SELECT td_clusterid_kmeans, 
                       wdname AS top_wdname
                FROM {feature_db}.{tables['sql_query_clusters']}
                GROUP BY td_clusterid_kmeans, wdname
                QUALIFY ROW_NUMBER() OVER (PARTITION BY td_clusterid_kmeans ORDER BY COUNT(*) DESC) = 1
            ) wd ON un.td_clusterid_kmeans = wd.td_clusterid_kmeans
            JOIN (
                SELECT td_clusterid_kmeans, 
                       appid AS top_AppId
                FROM {feature_db}.{tables['sql_query_clusters']}
                GROUP BY td_clusterid_kmeans, appid
                QUALIFY ROW_NUMBER() OVER (PARTITION BY td_clusterid_kmeans ORDER BY COUNT(*) DESC) = 1
            ) ap ON un.td_clusterid_kmeans = ap.td_clusterid_kmeans
            CROSS JOIN (
                SELECT * FROM TD_Silhouette(
                    ON (SELECT td_clusterid_kmeans, b.* 
                        FROM {feature_db}.{tables['sql_query_clusters']} a 
                        JOIN {feature_db}.{tables['sql_log_embeddings_store']} b
                        ON a.id = b.id) AS InputTable
                    USING
                        IdColumn('id')
                        ClusterIdColumn('td_clusterid_kmeans')
                        TargetColumns('[4:]')
                        OutputType('SCORE')
                ) AS dt
            ) AS s1
            JOIN (
                SELECT * FROM TD_Silhouette(
                    ON (SELECT td_clusterid_kmeans, b.* 
                        FROM {feature_db}.{tables['sql_query_clusters']} a 
                        JOIN {feature_db}.{tables['sql_log_embeddings_store']} b
                        ON a.id = b.id) AS InputTable
                    USING
                        IdColumn('id')
                        ClusterIdColumn('td_clusterid_kmeans')
                        TargetColumns('[4:]')
                        OutputType('CLUSTER_SCORES')
                ) AS dt
            ) s2 ON a.td_clusterid_kmeans = s2.td_clusterid_kmeans
            GROUP BY a.td_clusterid_kmeans
        ) WITH DATA PRIMARY INDEX(td_clusterid_kmeans)
        """
        
        cur.execute(cluster_stats_sql)
        logger.debug(f"Created cluster statistics table")

        # Get final results
        cur.execute(f"SELECT COUNT(*) FROM {feature_db}.{tables['sql_query_clusters']}")
        total_queries = cur.fetchone()[0]
        
        cur.execute(f"SELECT COUNT(DISTINCT td_clusterid_kmeans) FROM {feature_db}.{tables['sql_query_clusters']}")
        total_clusters = cur.fetchone()[0]
        
        cur.execute(f"SELECT AVG(silhouette_score) FROM {feature_db}.{tables['sql_query_clusters']}")
        avg_silhouette = cur.fetchone()[0]

    # Return comprehensive metadata
    metadata = {
        "tool_name": "sql_clustering_executeFullPipeline",
        "workflow_steps": [
            "query_log_extracted", "queries_tokenized", "embeddings_generated", 
            "embeddings_stored", "kmeans_clustering_completed", "silhouette_scores_calculated", 
            "cluster_statistics_generated"
        ],
        "configuration": {
            "optimal_k": optimal_k,
            "max_queries_processed": max_queries,
            "model_id": model_id,
            "clustering_parameters": clustering_config,
            "embedding_parameters": embedding_config
        },
        "results": {
            "total_queries_clustered": total_queries,
            "total_clusters_created": total_clusters,
            "average_silhouette_score": float(avg_silhouette) if avg_silhouette else None
        },
        "tables_created": [
            f"{feature_db}.{tables['sql_query_log_main']}",
            f"{feature_db}.{tables['sql_log_tokenized_for_embeddings']}",
            f"{feature_db}.{tables['sql_log_embeddings']}",
            f"{feature_db}.{tables['sql_log_embeddings_store']}",
            f"{feature_db}.{tables['sql_query_clusters']}",
            f"{feature_db}.{tables['query_cluster_stats']}"
        ],
        "description": "Complete SQL query clustering pipeline executed: extracted SQL logs → tokenized → embedded → clustered → analyzed"
    }

    return create_response({"status": "success", "pipeline_completed": True}, metadata)


def handle_sql_clustering_analyzeClusterStats(
    conn,
    sort_by_metric: str = "avg_cpu",
    limit_results: int = None,
    *args,
    **kwargs
):
    """
    Analyze existing cluster statistics without re-running clustering.
    
    This function reads the pre-computed cluster statistics table and provides
    analysis of query clusters sorted by the specified performance metric.
    
    Returns cluster analysis data for LLM to suggest optimization focus areas.
    """
    
    config = SQL_CLUSTERING_CONFIG
    
    logger.debug(f"handle_sql_clustering_analyzeClusterStats: sort_by={sort_by_metric}, limit={limit_results}")
    
    feature_db = config['databases']['feature_db']
    stats_table = config['tables']['query_cluster_stats']
    
    # Validate sort metric
    valid_metrics = [
        'avg_cpu', 'avg_io', 'avg_cpuskw', 'avg_ioskw', 'avg_pji', 'avg_uii',
        'avg_numsteps', 'queries', 'cluster_silhouette_score'
    ]
    
    if sort_by_metric not in valid_metrics:
        sort_by_metric = 'avg_cpu'  # Default fallback

    with conn.cursor() as cur:
        
        # Build the query with optional limit
        limit_clause = f"TOP {limit_results}" if limit_results else ""
        
        # Get thresholds from config
        thresholds = config.get('performance_thresholds', {})
        cpu_high = thresholds.get('cpu', {}).get('high', 100)
        skew_high = thresholds.get('skew', {}).get('high', 3.0)
        io_high = thresholds.get('io', {}).get('high', 1000000)

        stats_query = f"""
        SELECT {limit_clause}
            td_clusterid_kmeans,
            avg_numsteps, 
            var_numsteps,
            avg_cpu, 
            var_cpu,
            avg_io, 
            var_io,
            avg_cpuskw, 
            var_cpuskw,
            avg_ioskw, 
            var_ioskw,
            avg_pji, 
            var_pji,
            avg_uii, 
            var_uii,
            top_username,
            top_wdname,
            top_appid,
            overall_silhouette_score,
            cluster_silhouette_score,
            queries,
            -- Additional analysis columns with configurable thresholds
            CASE 
                WHEN avg_cpuskw > {skew_high} THEN 'HIGH_CPU_SKEW'
                WHEN avg_ioskw > {skew_high} THEN 'HIGH_IO_SKEW'
                WHEN avg_cpu > {cpu_high} THEN 'HIGH_CPU_USAGE'
                WHEN avg_io > {io_high} THEN 'HIGH_IO_USAGE'
                ELSE 'NORMAL'
            END AS performance_category,
            RANK() OVER (ORDER BY {sort_by_metric} DESC) AS performance_rank
        FROM {feature_db}.{stats_table}
        ORDER BY {sort_by_metric} DESC
        """
        
        cur.execute(stats_query)
        data = rows_to_json(cur.description, cur.fetchall())
        
        # Get summary statistics
        cur.execute(f"""
        SELECT 
            COUNT(*) AS total_clusters,
            AVG(avg_cpu) AS system_avg_cpu,
            AVG(avg_io) AS system_avg_io,
            AVG(queries) AS avg_queries_per_cluster,
            MAX(avg_cpu) AS max_cluster_cpu,
            MIN(cluster_silhouette_score) AS min_silhouette_score
        FROM {feature_db}.{stats_table}
        """)
        
        summary_stats = rows_to_json(cur.description, cur.fetchall())[0]
        
        logger.debug(f"Retrieved {len(data)} cluster statistics")

    # Return results with comprehensive metadata
    metadata = {
        "tool_name": "sql_clustering_analyzeClusterStats",
        "analysis_parameters": {
            "sort_by_metric": sort_by_metric,
            "limit_results": limit_results,
            "valid_metrics": valid_metrics
        },
        "summary_statistics": summary_stats,
        "clusters_analyzed": len(data),
        "table_source": f"{feature_db}.{stats_table}",
        "description": f"Cluster statistics analysis sorted by {sort_by_metric} - ready for LLM optimization recommendations"
    }

    return create_response(data, metadata)


def handle_sql_clustering_retrieveClusterQueries(
    conn,
    cluster_ids: List[int],
    metric: str = "ampcputime",
    limit_per_cluster: int = 250,
    *args,
    **kwargs
):
    """
    Retrieve actual SQL queries from specified clusters for pattern analysis and optimization.
    
    This function extracts the top queries from selected clusters based on a performance metric,
    allowing the LLM to analyze actual SQL patterns and propose specific optimizations.
    
    Returns actual SQL queries with performance metrics for detailed analysis.
    """
    
    config = SQL_CLUSTERING_CONFIG
    
    logger.debug(f"handle_sql_clustering_retrieveClusterQueries: clusters={cluster_ids}, metric={metric}, limit={limit_per_cluster}")
    
    feature_db = config['databases']['feature_db']
    clusters_table = config['tables']['sql_query_clusters']
    
    # Validate metric
    valid_metrics = [
        'ampcputime', 'logicalio', 'cpuskw', 'ioskw', 'pji', 'uii',
        'numsteps', 'response_secs', 'delaytime'
    ]
    
    if metric not in valid_metrics:
        metric = 'ampcputime'  # Default fallback

    # Convert cluster_ids list to comma-separated string for SQL IN clause
    cluster_ids_str = ','.join(map(str, cluster_ids))

    with conn.cursor() as cur:
        
        # Get thresholds from config
        thresholds = config.get('performance_thresholds', {})
        cpu_high = thresholds.get('cpu', {}).get('high', 100)
        cpu_very_high = thresholds.get('cpu', {}).get('very_high', 1000)
        skew_moderate = thresholds.get('skew', {}).get('moderate', 2.0)
        skew_high = thresholds.get('skew', {}).get('high', 3.0)
        skew_severe = thresholds.get('skew', {}).get('severe', 5.0)
        
        retrieve_queries_sql = f"""
        SELECT 
            td_clusterid_kmeans,
            id,
            txt,
            username,
            appid,
            numsteps,
            ampcputime,
            logicalio,
            wdname,
            cpuskw,
            ioskw,
            pji,
            uii,
            response_secs,
            response_mins,
            delaytime,
            silhouette_score,
            -- Ranking within cluster by selected metric
            ROW_NUMBER() OVER (PARTITION BY td_clusterid_kmeans ORDER BY {metric} DESC) AS rank_in_cluster,
            -- Overall ranking across all selected clusters
            ROW_NUMBER() OVER (ORDER BY {metric} DESC) AS overall_rank,
            -- Performance categorization with configurable thresholds
            CASE 
                WHEN ampcputime > {cpu_very_high} THEN 'VERY_HIGH_CPU'
                WHEN ampcputime > {cpu_high} THEN 'HIGH_CPU'
                WHEN ampcputime > 10 THEN 'MEDIUM_CPU'
                ELSE 'LOW_CPU'
            END AS cpu_category,
            CASE 
                WHEN cpuskw > {skew_severe} THEN 'SEVERE_CPU_SKEW'
                WHEN cpuskw > {skew_high} THEN 'HIGH_CPU_SKEW'
                WHEN cpuskw > {skew_moderate} THEN 'MODERATE_CPU_SKEW'
                ELSE 'NORMAL_CPU_SKEW'
            END AS cpu_skew_category,
            CASE 
                WHEN ioskw > {skew_severe} THEN 'SEVERE_IO_SKEW'
                WHEN ioskw > {skew_high} THEN 'HIGH_IO_SKEW'
                WHEN ioskw > {skew_moderate} THEN 'MODERATE_IO_SKEW'
                ELSE 'NORMAL_IO_SKEW'
            END AS io_skew_category
        FROM {feature_db}.{clusters_table}
        WHERE td_clusterid_kmeans IN ({cluster_ids_str})
        QUALIFY ROW_NUMBER() OVER (PARTITION BY td_clusterid_kmeans ORDER BY {metric} DESC) <= {limit_per_cluster}
        ORDER BY td_clusterid_kmeans, {metric} DESC
        """
        
        cur.execute(retrieve_queries_sql)
        data = rows_to_json(cur.description, cur.fetchall())
        
        # Get summary by cluster
        cur.execute(f"""
        SELECT 
            td_clusterid_kmeans,
            COUNT(*) AS queries_retrieved,
            AVG({metric}) AS avg_metric_value,
            MAX({metric}) AS max_metric_value,
            MIN({metric}) AS min_metric_value
        FROM {feature_db}.{clusters_table}
        WHERE td_clusterid_kmeans IN ({cluster_ids_str})
        GROUP BY td_clusterid_kmeans
        ORDER BY td_clusterid_kmeans
        """)
        
        cluster_summary = rows_to_json(cur.description, cur.fetchall())
        
        logger.debug(f"Retrieved {len(data)} queries from {len(cluster_ids)} clusters")

    # Return results with comprehensive metadata
    metadata = {
        "tool_name": "sql_clustering_retrieveClusterQueries",
        "retrieval_parameters": {
            "cluster_ids": cluster_ids,
            "sort_metric": metric,
            "limit_per_cluster": limit_per_cluster,
            "valid_metrics": valid_metrics
        },
        "cluster_summary": cluster_summary,
        "queries_retrieved": len(data),
        "table_source": f"{feature_db}.{clusters_table}",
        "analysis_ready": True,
        "description": f"Retrieved top {limit_per_cluster} queries per cluster sorted by {metric} - ready for pattern analysis and optimization recommendations"
    }

    return create_response(data, metadata)






######################################################################################
# Financial Report Analysis Tool
######################################################################################

import logging
import yaml
from typing import Optional, Any, Dict, List
from teradatasql import TeradataConnection
import json
from datetime import date, datetime
from decimal import Decimal

logger = logging.getLogger("teradata_mcp_server")

# Load Financial RAG configuration
def load_financial_rag_config():
    """Load Financial RAG configuration from financial_rag_config.yaml"""
    try:
        with open('financial_rag_config.yaml', 'r') as file:
            logger.info("Loading Financial RAG config from: financial_rag_config.yaml")
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning("Financial RAG config file not found: financial_rag_config.yaml, using defaults")
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
    
    # Always add similarity (calculated field)
    select_fields.append("(1.0 - dt.distance) AS similarity")
    
    select_clause = ",\n            ".join(select_fields)
    
    # Build WHERE clause
    where_conditions = [f"(1.0 - dt.distance) >= {min_similarity}"]
    
    # Add year filtering if years specified - FIXED to use report_year
    if years:
        year_list = ','.join(map(str, years))
        where_conditions.append(f"e_ref.report_year IN ({year_list})")
    
    where_clause = " AND ".join(where_conditions)
    
    # Multi-year strategy: get balanced chunks per year
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

def serialize_teradata_types(obj: Any) -> Any:
    """Convert Teradata-specific types to JSON serializable formats"""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)

def rows_to_json(cursor_description: Any, rows: List[Any]) -> List[Dict[str, Any]]:
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

def create_response(data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Create a standardized JSON response structure"""
    response = {
        "status": "success",
        "metadata": metadata or {},
        "results": data
    }
    return json.dumps(response, default=serialize_teradata_types)

def handle_financial_rag_analysis(
    conn: TeradataConnection,
    question: str,
    years: List[int] = None,
    analysis_type: str = "general",
    k: int = None,
    *args,
    **kwargs,
):
    """
    Execute financial reports RAG analysis with LLM-determined parameters for ICICI Bank annual reports.
    
    This function performs intelligent retrieval based on LLM-parsed parameters:
    
    Args:
        question: User's financial analysis question
        years: List of specific years to analyze (e.g., [2009, 2010, 2011] for multi-year analysis)
        analysis_type: Type of analysis - "temporal" (trends over time), "comparative" (compare periods), "general" (standard)
        k: Total number of chunks to retrieve (uses smart defaults based on years if None)
    
    Returns:
        JSON response with retrieved chunks optimized for the analysis type
    """
    
    config = FINANCIAL_RAG_CONFIG
    
    logger.debug(f"Financial RAG Analysis: question={question[:60]}...")
    logger.debug(f"Parameters: years={years}, analysis_type={analysis_type}")
    
    # Smart defaults based on analysis type and parameters
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
        
        # Step 1: Store user query with metadata
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

        # Step 2: Generate query embeddings using IVSM
        logger.debug("Generating embeddings using IVSM pipeline")
        
        # Step 2a: Tokenize query (EXACT syntax from working function)
        logger.debug("Step 2a: Tokenizing query using ivsm.tokenizer_encode")
        
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

        # Step 2b: Create embedding view (EXACT syntax from working function)
        logger.debug("Step 2b: Creating embedding view using ivsm.IVSM_score")
        
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

        # Step 2c: Create query embedding table (EXACT syntax from working function)
        logger.debug("Step 2c: Creating query embedding table using ivsm.vector_to_columns")
        
        # Drop existing embeddings table
        drop_sql = f"DROP TABLE {db_name}.{dst_table}"
        try:
            cur.execute(drop_sql)
            logger.debug(f"Dropped existing table {db_name}.{dst_table}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        # Create embeddings table using vector_to_columns (CORRECTED)
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

        # Step 3: Perform semantic search with year filtering
        logger.debug("Step 3: Performing filtered semantic search")
        
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

    # Return structured metadata
    metadata = {
        "tool_name": "financial_rag_analysis",
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
        "description": f"ICICI Bank financial analysis ({analysis_type}) with LLM-determined parameters"
    }

    return create_response(data, metadata)