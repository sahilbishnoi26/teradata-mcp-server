import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any
from pathlib import Path

import yaml
from teradatasql import TeradataConnection

logger = logging.getLogger("complaint_similarity_server")

# Load complaint similarity configuration
def load_complaint_config():
    """Load complaint similarity configuration from complaint_config.yml"""
    try:
        # Get the directory path
        current_dir = Path(__file__).parent
        # Go to config/
        config_path = current_dir.parent.parent / 'config' / 'complaint_config.yml'
        
        with open(config_path, 'r') as file:
            logger.info(f"Loading complaint config from: {config_path}")
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning(f"Complaint config file not found: {config_path}, using defaults")
        return get_default_complaint_config()
    except Exception as e:
        logger.error(f"Error loading complaint config: {e}")
        return get_default_complaint_config()

def get_default_complaint_config():
    """Default complaint similarity configuration as fallback"""
    return {
        'databases': {
            'query_db': 'demo_db',
            'model_db': 'demo_db',
            'complaint_db': 'demo_db'
        },
        'tables': {
            'query_table': 'complaint_query',
            'query_embedding_store': 'complaint_query_embeddings',
            'complaints_table': 'wf_complaints_master',
            'model_table': 'embeddings_models',
            'tokenizer_table': 'embeddings_tokenizers'
        },
        'model': {
            'model_id': 'bge-small-en-v1.5'
        },
        'retrieval': {
            'default_k': 5,
            'max_k': 20
        },
        'complaint_fields': {
            'required_fields': ['complaint_id', 'description_text'],
            'metadata_fields': [
                'customer_id', 'issue_cat_id', 'opened_ts', 'closed_ts', 
                'status', 'severity', 'priority', 'agent_notes', 
                'resolution_code', 'refund_amount', 'emotion_primary'
            ]
        },
        'embedding': {
            'vector_length': 384,
            'vector_column_prefix': 'emb_',
            'distance_measure': 'cosine',
            'feature_columns': '[emb_0:emb_383]'
        }
    }

# Load config at module level
COMPLAINT_CONFIG = load_complaint_config()

def build_complaint_search_query(complaint_db, dst_table, complaints_table, k, config):
    """Build complaint similarity search query using integer IDs"""
    # Get metadata fields from config
    metadata_fields = config['complaint_fields']['metadata_fields'] or []
    feature_columns = config['embedding']['feature_columns']

    # Build SELECT clause dynamically - complaint_id and description_text are always required
    select_fields = [
        "c.complaint_id",
        "c.description_text"
    ]

    # Add all metadata fields from complaints table
    for field in metadata_fields:
        if field not in ['complaint_id', 'description_text']:
            select_fields.append(f"c.{field}")

    # Add similarity
    select_fields.append("(1.0 - dt.distance) AS similarity")

    select_clause = ",\n            ".join(select_fields)

    return f"""
        SELECT
            {select_clause}
        FROM TD_VECTORDISTANCE (
                ON {complaint_db}.{dst_table}      AS TargetTable
                ON {complaint_db}.{complaints_table}      AS ReferenceTable DIMENSION
                USING
                    TargetIDColumn('id')
                    TargetFeatureColumns('{feature_columns}')
                    RefIDColumn('id')
                    RefFeatureColumns('{feature_columns}')
                    DistanceMeasure('cosine')
                    TopK({k})
            ) AS dt
        JOIN {complaint_db}.{complaints_table} c
          ON c.id = dt.reference_id
        ORDER BY similarity DESC;
        """

def serialize_teradata_types(obj: Any) -> Any:
    """Convert Teradata-specific types to JSON serializable formats"""
    if isinstance(obj, (datetime,)):
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

def handle_complaint_similarity_search(
    conn: TeradataConnection,
    description: str,
    k: int | None = None,
    *args,
    **kwargs,
):
    """
    **COMPLAINT SIMILARITY SEARCH FOR CASE WORKER SUPPORT**

    Find similar complaints from description text using semantic embeddings.  
    The goal is to help case workers understand patterns, learn effective resolution strategies, and provide context-driven support.

    ---

    ### WHEN TO USE
    - A new complaint arrives and you want to see how similar cases were handled  
    - A case worker needs guidance from past resolutions before advising a customer  
    - Customer escalation requires precedent from similar issues  
    - Training new staff on common complaint categories and approaches  
    - Quality assurance review for consistency of complaint handling  
    - Supervisors assessing case complexity based on historical precedent  

    **Example scenarios**
    - "Customer’s debit card replacement hasn’t arrived after 10 days" → find similar card replacement delays  
    - "Overdraft fees charged during system maintenance" → find similar fee disputes during outages  
    - "Mobile app login failing after password reset" → find similar digital banking authentication issues  
    - "Wire transfer held up for international payment" → find similar wire transfer compliance delays  
    - "ATM dispensed wrong amount during withdrawal" → find similar ATM discrepancy cases  

    ---

    ### TECHNICAL WORKFLOW
    1. Store the search description as a query  
    2. Generate embeddings using BYOM (ONNXEmbeddings)  
    3. Perform semantic similarity search against historical complaints  
    4. Return the top-k most similar complaints with relevant metadata  

    **Args**
    - `description`: text description of the complaint to search  
    - `k`: number of similar complaints to return (default 5, max 20)  

    **Returns**
    JSON response with the most relevant complaints, including:  
    - complaint_id: unique identifier (e.g., "CMP000001")  
    - description_text: original complaint description  
    - similarity: cosine similarity score (0–1, higher = more similar)  
    - resolution_code: how the case was resolved  
    - refund_amount: compensation if applicable  
    - agent_notes: resolution steps taken  
    - severity, priority, status: case classification and state  
    - opened_ts, closed_ts: complaint timeline  
    - emotion_primary: primary customer emotion  
    - other metadata fields: customer_id, escalation indicators, etc.  

    ---

    ### LLM RESPONSE GUIDELINES FOR CASE WORKER SUPPORT

    #### Strong Similarity (similarity ≥ 0.7)
    - Title the section: *"Strongly similar complaints found"*  
    - List 3–5 matches with: complaint_id, one-line description, resolution_code, typical timeline, refund_amount if present  
    - Summarize common resolution patterns (from resolution_code and agent_notes)  
    - Highlight pitfalls and escalation triggers seen in similar cases  
    - Provide 1–2 clear, actionable next steps  

    #### Moderate Similarity (0.4–0.7)
    - Title the section: *"Related but not exact matches"*  
    - Call out what is similar and what differs  
    - Suggest which resolution strategies might transfer, with caveats  
    - Provide careful recommendations and checks to run first  

    #### No Similar Cases (similarity < 0.4 or results empty)
    - Title the section: *"No closely matching complaints found"*  
    - Do **not** list complaints  
    - Provide general best-practice guidance for the case type:  
    - risk controls (e.g., blocking card, resetting login)  
    - investigation checklist  
    - communication plan and customer updates  
    - escalation triggers and documentation requirements  
    - typical resolution timelines by category  
    - Be explicit: "Since no similar cases were found, these recommendations are based on general best practices rather than precedent."  

    ---

    ### ALWAYS INCLUDE
    - Actionable next steps for the case worker  
    - Timeline expectations (based on data or best practices)  
    - Customer communication suggestions  
    - Documentation needs  
    - Escalation triggers to monitor  
    - Success criteria or resolution confirmation steps  

    ---

    ### CONTEXT-DRIVEN INSIGHTS
    Leverage complaint metadata to give case workers practical takeaways, for example:  
    - "Similar expedited card cases typically resolve in 2–3 business days"  
    - "90% of these cases resulted in fee waivers of $25–35"  
    - "Escalations are common if the customer mentions business travel urgency"  
    - "Proactive status updates every 24 hours led to higher satisfaction"  

    ---

    **End goal:** transform complaint similarity results into concise, actionable guidance that improves resolution efficiency and customer experience.
    """

    
    # Use configuration from loaded config
    config = COMPLAINT_CONFIG
    
    # Use config default if k not provided
    if k is None:
        k = config['retrieval']['default_k']

    # Optional: Enforce max limit
    max_k = config['retrieval'].get('max_k', 20)
    if k > max_k:
        logger.warning(f"Requested k={k} exceeds max_k={max_k}, using max_k")
        k = max_k

    logger.debug(f"handle_complaint_similarity_search: description={description[:60]}..., k={k}")

    # Extract config values
    database_name = config['databases']['query_db']
    table_name = config['tables']['query_table']
    dst_table = config['tables']['query_embedding_store']
    model_id = config['model']['model_id']
    model_db = config['databases']['model_db']
    model_table = config['tables']['model_table']
    tokenizer_table = config['tables']['tokenizer_table']
    complaint_db = config['databases']['complaint_db']
    complaints_table = config['tables']['complaints_table']

    with conn.cursor() as cur:
        # Store search query
        logger.debug(f"Step 1: Storing search query in {database_name}.{table_name}")

        # Create table if it doesn't exist
        ddl = f"""
        CREATE TABLE {database_name}.{table_name} (
            id INTEGER GENERATED ALWAYS AS IDENTITY (START WITH 1 INCREMENT BY 1) NOT NULL,
            txt VARCHAR(5000),
            created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id)
        )
        """

        try:
            cur.execute(ddl)
            logger.debug(f"Table {database_name}.{table_name} created")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "3803" in error_msg:
                logger.debug(f"Table {database_name}.{table_name} already exists, skipping creation")
            else:
                logger.error(f"Error creating table: {e}")
                raise

        # Insert search description
        insert_sql = f"INSERT INTO {database_name}.{table_name} (txt) VALUES (?)"
        cur.execute(insert_sql, [description.strip()])

        # Get inserted ID
        cur.execute(f"SELECT MAX(id) AS id FROM {database_name}.{table_name}")
        new_id = cur.fetchone()[0]

        logger.debug(f"Stored search query with ID {new_id}: {description[:60]}...")

        # Generate query embeddings
        logger.debug(f"Step 2: Generating embeddings in {database_name}.{dst_table}")

        # Drop existing embeddings table
        drop_sql = f"DROP TABLE {database_name}.{dst_table}"
        try:
            cur.execute(drop_sql)
            logger.debug(f"Dropped existing table {database_name}.{dst_table}")
        except Exception as e:
            logger.debug(f"DROP failed or table not found: {e}")

        # Create embeddings table with integer ID
        create_sql = f"""
        CREATE TABLE {database_name}.{dst_table} AS (
            SELECT *
            FROM mldb.ONNXEmbeddings(
                ON (SELECT id, txt FROM {database_name}.{table_name} WHERE id = {new_id})
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
        logger.debug(f"Created embeddings table {database_name}.{dst_table}")

        # Perform similarity search
        logger.debug(f"Step 3: Performing similarity search with k={k}")

        search_sql = build_complaint_search_query(complaint_db, dst_table, complaints_table, k, config)

        rows = cur.execute(search_sql)
        data = rows_to_json(cur.description, rows.fetchall())

        logger.debug(f"Retrieved {len(data)} similar complaints")

    # Return metadata
    metadata = {
        "tool_name": "complaint_similarity_search",
        "search_description": description.strip(),
        "query_id": new_id,
        "database": database_name,
        "query_table": table_name,
        "embedding_table": dst_table,
        "complaints_table": complaints_table,
        "model_id": model_id,
        "similar_complaints_found": len(data),
        "topk_requested": k,
        "topk_configured_default": config['retrieval']['default_k'],
        "metadata_fields_returned": config['complaint_fields']['metadata_fields'],
        "description": "Semantic similarity search for complaints using BYOM embeddings with integer ID compatibility"
    }
    
    logger.debug(f"Tool: handle_complaint_similarity_search: metadata: {metadata}")
    return create_response(data, metadata)





#########################################################################################################

import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, List
from pathlib import Path

import yaml
from teradatasql import TeradataConnection

logger = logging.getLogger("interaction_history_server")

# Load interaction history configuration
def load_interaction_config():
    """Load interaction history configuration from interaction_config.yml"""
    try:
        current_dir = Path(__file__).parent
        config_path = current_dir.parent.parent / 'config' / 'interaction_config.yml'
        
        with open(config_path, 'r') as file:
            logger.info(f"Loading interaction config from: {config_path}")
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning(f"Interaction config file not found: {config_path}, using defaults")
        return get_default_interaction_config()
    except Exception as e:
        logger.error(f"Error loading interaction config: {e}")
        return get_default_interaction_config()

def get_default_interaction_config():
    """Default interaction history configuration as fallback"""
    return {
        'databases': {
            'interaction_db': 'demo_db',
            'complaint_db': 'demo_db'
        },
        'tables': {
            'interactions_table': 'wf_interactions',
            'complaints_table': 'wf_complaints'
        },
        'retrieval': {
            'default_limit': 50,
            'max_limit': 200
        },
        'interaction_fields': {
            'required_fields': [
                'interaction_id', 'complaint_id', 'customer_id', 
                'interaction_ts', 'channel', 'agent_id', 
                'summary_text', 'disposition', 'sentiment_score'
            ],
            'detail_fields': [
                'full_transcript_text', 'agent_private_notes', 
                'customer_visible_notes', 'duration_seconds',
                'follow_up_required', 'follow_up_due_ts', 'priority',
                'satisfaction_predicted', 'resolution_confidence',
                'call_quality_score', 'key_topics_extracted',
                'next_best_action', 'customer_effort_predicted'
            ]
        },
        'sorting': {
            'default_order': 'interaction_ts ASC',
            'available_orders': [
                'interaction_ts ASC', 'interaction_ts DESC',
                'sentiment_score DESC', 'priority DESC',
                'satisfaction_predicted ASC', 'duration_seconds DESC'
            ]
        }
    }

# Load config at module level
INTERACTION_CONFIG = load_interaction_config()

def serialize_teradata_types(obj: Any) -> Any:
    """Convert Teradata-specific types to JSON serializable formats"""
    if isinstance(obj, (datetime,)):
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

def handle_complaint_interaction_history(
    conn: TeradataConnection,
    complaint_ids: List[str],
    limit_per_complaint: int = None,
    order_by: str = None,
    include_details: bool = True,
    *args,
    **kwargs,
):
    """
    **COMPLAINT INTERACTION HISTORY FOR CASE WORKER SUPPORT**

    Retrieve and summarize interaction history for one or more complaints.  
    Goal: give case workers a clear, actionable brief that prevents customers from repeating themselves and highlights what matters next.

    ---

    ### WHEN TO USE
    - During ongoing complaints to give agents full context  
    - At handoff to a new case worker  
    - For complex escalations or supervisor reviews  
    - For callback preparation  
    - To study how similar complaints were resolved  

    **Example scenarios**
    - "Customer calling back about CMP000003 — what did we discuss yesterday?"  
    - "Need history for CMP000015 before escalating"  
    - "Show me how we handled similar card replacement cases"  
    - "Customer says they already explained the issue — what did they tell the previous agent?"  

    ---

    ### CONTEXT EXTRACTION
    This tool can resolve complaint IDs in two ways:

    1. **From the current query** – infer IDs from natural language:  
    - "show me interactions for this complaint" → uses ID from current case context  
    - "what happened in the gym fee case?" → maps phrase to complaint ID  

    2. **From conversation history** – reuse IDs from prior responses:  
    - "pull history for the similar case" → uses ID from recent similarity search results  
    - "those card replacement complaints" → extracts multiple IDs from history  
    - "that case we just found" → reuses complaint ID from last tool output  

    ---

    ### OUTPUT FORMAT
    Responses should be **case-worker friendly briefs** with these sections:

    1. **case at a glance** – complaint_id, status, severity, one-line issue, key dates, customer risk  
    2. **what the customer already told us** – bullets of facts already provided  
    3. **timeline** – one line per interaction:  
    `YYYY-MM-DD • Channel • 6–12 words • disposition`  
    4. **promises and deadlines** – commitments with dates/timeframes  
    5. **sentiment trend** – simple trend like `negative → neutral → positive`  
    6. **open items** – tasks, risks, or verifications needed  
    7. **use this tone** – one sentence on tone/approach, include channel preference if noted  
    8. **next best action** – 1–2 specific, actionable bullets  

    ---

    ### STYLE RULES
    - Use short sentences and plain words  
    - Prefer bullets over paragraphs  
    - Collapse repeated complaint descriptions  
    - Show dates as `YYYY-MM-DD`  
    - Channels limited to: Phone, Chat, Email, SMS, Branch, Secure Message  
    - Only include metrics if they affect action (e.g., very low satisfaction)  
    - Skip transcripts and raw rows unless explicitly requested  

    ---

    ### RETURNS
    JSON response containing the structured brief plus metadata.  

    **End goal:** transform raw history into an actionable case summary that ensures continuity of service and equips agents with clear next steps.
    """

    
    # Use configuration
    config = INTERACTION_CONFIG
    
    # Set defaults
    if limit_per_complaint is None:
        limit_per_complaint = config['retrieval']['default_limit']
    
    # Enforce max limit
    max_limit = config['retrieval']['max_limit']
    if limit_per_complaint > max_limit:
        logger.warning(f"Requested limit {limit_per_complaint} exceeds max {max_limit}, using max")
        limit_per_complaint = max_limit
    
    if order_by is None:
        order_by = config['sorting']['default_order']
    
    # Validate order_by
    if order_by not in config['sorting']['available_orders']:
        logger.warning(f"Invalid order_by {order_by}, using default")
        order_by = config['sorting']['default_order']
    
    logger.debug(f"handle_complaint_interaction_history: complaints={complaint_ids}, limit={limit_per_complaint}")

    # Extract config values  
    interaction_db = config['databases']['interaction_db']
    complaint_db = config['databases']['complaint_db']
    interactions_table = config['tables']['interactions_table']
    complaints_table = config['tables']['complaints_table']
    
    # Build dynamic SELECT clause
    required_fields = config['interaction_fields']['required_fields']
    detail_fields = config['interaction_fields']['detail_fields'] if include_details else []
    
    all_fields = required_fields + detail_fields
    select_fields = [f"i.{field}" for field in all_fields]
    
    # Add complaint context
    select_fields.extend([
        "c.description_text", 
        "c.status as complaint_status",
        "c.severity",
        "c.resolution_code"
    ])
    
    select_clause = ",\n            ".join(select_fields)
    
    # Build WHERE clause for multiple complaint IDs
    complaint_placeholders = ', '.join(['?' for _ in complaint_ids])
    
    # Build limit clause - limit per complaint using window functions
    if len(complaint_ids) == 1:
        # Single complaint - use TOP instead of LIMIT for Teradata
        top_clause = f"TOP {limit_per_complaint}"
        qualify_clause = ""
        limit_clause = ""
    else:
        # Multiple complaints - limit per complaint using QUALIFY
        top_clause = ""
        qualify_clause = f"QUALIFY ROW_NUMBER() OVER (PARTITION BY i.complaint_id ORDER BY i.{order_by.replace(' ASC', '').replace(' DESC', '')}) <= {limit_per_complaint}"
        limit_clause = ""

    with conn.cursor() as cur:
        # Build and execute query
        query = f"""
        SELECT {top_clause}
            {select_clause}
        FROM {interaction_db}.{interactions_table} i
        LEFT JOIN {complaint_db}.{complaints_table} c
            ON i.complaint_id = c.complaint_id  
        WHERE i.complaint_id IN ({complaint_placeholders})
        {qualify_clause}
        ORDER BY i.complaint_id, i.{order_by}
        """
        
        logger.debug(f"Executing query: {query}")
        logger.debug(f"Parameters: {complaint_ids}")
        
        cur.execute(query, complaint_ids)
        data = rows_to_json(cur.description, cur.fetchall())
        
        logger.debug(f"Retrieved {len(data)} interactions for {len(complaint_ids)} complaints")

    # Return metadata
    metadata = {
        "tool_name": "complaint_interaction_history",
        "complaint_ids": complaint_ids,
        "interactions_found": len(data),
        "limit_per_complaint": limit_per_complaint,
        "order_by": order_by,
        "include_details": include_details,
        "database": interaction_db,
        "interactions_table": interactions_table,
        "fields_returned": len(all_fields) + 4,  # +4 for complaint context fields
        "description": "Complete interaction history retrieval for complaint context and case worker support"
    }
    
    logger.debug(f"Tool: handle_complaint_interaction_history: metadata: {metadata}")
    return create_response(data, metadata)






##########################################################################################################


import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from pathlib import Path

import yaml
from teradatasql import TeradataConnection

logger = logging.getLogger("customer_summary_server")

def load_customer_summary_config():
    """Load customer summary configuration from customer_summary_config.yml"""
    try:
        current_dir = Path(__file__).parent
        config_path = current_dir.parent.parent / 'config' / 'customer_summary_config.yml'
        
        with open(config_path, 'r') as file:
            logger.info(f"Loading customer summary config from: {config_path}")
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning(f"Customer summary config file not found: {config_path}, using defaults")
        return get_default_customer_summary_config()
    except Exception as e:
        logger.error(f"Error loading customer summary config: {e}")
        return get_default_customer_summary_config()

def get_default_customer_summary_config():
    """Default customer summary configuration as fallback"""
    return {
        'databases': {
            'customer_db': 'demo_db',
            'account_db': 'demo_db', 
            'transaction_db': 'demo_db',
            'complaint_db': 'demo_db',
            'interaction_db': 'demo_db'
        },
        'tables': {
            'customers_table': 'wf_customer_profiles',
            'accounts_table': 'wf_accounts',
            'transactions_table': 'wf_transactions',
            'complaints_table': 'wf_complaints',
            'interactions_table': 'wf_interactions'
        },
        'analysis': {
            'recent_transaction_days': 30,
            'complaint_lookback_days': 180,
            'interaction_lookback_days': 90,
            'high_value_threshold': 150000,
            'churn_risk_high_threshold': 0.4,
            'complaint_propensity_high_threshold': 0.5,
            'product_recommendations': {
                'checking_balance_threshold': 10000,
                'savings_balance_threshold': 50000,
                'credit_utilization_low': 0.1,
                'credit_utilization_high': 0.8
            }
        },
        'risk_scoring': {
            'churn_factors': {
                'complaint_weight': 0.3,
                'interaction_sentiment_weight': 0.2,
                'financial_stress_weight': 0.2,
                'tenure_weight': 0.1,
                'engagement_weight': 0.2
            }
        }
    }

# Load config at module level
CUSTOMER_SUMMARY_CONFIG = load_customer_summary_config()

def serialize_teradata_types(obj: Any) -> Any:
    """Convert Teradata-specific types to JSON serializable formats"""
    if isinstance(obj, (datetime,)):
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

def safe_float(value, default=0.0):
    """Safely convert a value to float, handling strings and None"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert a value to int, handling strings and None"""
    if value is None:
        return default
    try:
        return int(float(value))  # Convert through float first to handle decimals
    except (ValueError, TypeError):
        return default

def safe_bool(value, default=False):
    """Safely convert a value to bool, handling strings and None"""
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'y')
    return bool(value)

def create_response(data: Any, metadata: dict[str, Any] | None = None) -> str:
    """Create a standardized JSON response structure"""
    response = {
        "status": "success",
        "metadata": metadata or {},
        "results": data
    }
    return json.dumps(response, default=serialize_teradata_types)

def handle_customer_summary_card(
    conn: TeradataConnection,
    customer_id: str,
    *args,
    **kwargs,
):
    """
    **COMPREHENSIVE CUSTOMER SUMMARY CARD FOR CASE WORKER SUPPORT**

    Generate a complete customer profile summary card that aggregates all relevant customer data 
    to provide case workers with instant context and actionable insights for personalized service.

    ---

    ### WHEN TO USE
    - Before any customer interaction to understand their full profile
    - During complaint handling to assess customer value and risk
    - When considering product recommendations or retention offers  
    - For escalation decisions and supervisor handoffs
    - To prepare for callback or follow-up conversations

    **Example scenarios**
    - "Pull up Sarah Chen's profile before I call her back"
    - "Need full context on this high-value customer before escalating"
    - "What products should I offer this customer?"
    - "Is this customer at risk of churning?"

    ---

    ### SUMMARY CARD SECTIONS

    #### 1. **Customer At-a-Glance**
    - Name, segment, tenure, VIP status
    - Customer value (LTV) and lifetime relationship summary
    - Primary contact preferences and branch affiliation

    #### 2. **Financial Profile**  
    - Account portfolio summary (checking, savings, credit, loans)
    - Current balances and credit utilization
    - Monthly spending patterns and transaction behavior
    - Financial health indicators and stress flags

    #### 3. **Risk Assessment**
    - Comprehensive churn risk score with contributing factors
    - Complaint propensity and escalation risk
    - Recent sentiment trends from interactions
    - Financial stress indicators and early warning signs

    #### 4. **Recent Activity**
    - Last 30 days transaction summary  
    - Open complaints and their status
    - Recent interactions and sentiment scores
    - Unusual activity or pattern changes

    #### 5. **Product Recommendations**
    - Personalized product suggestions based on profile
    - Cross-sell and upsell opportunities
    - Service improvements and feature adoptions
    - Revenue enhancement potential

    #### 6. **Case Worker Guidance**
    - Recommended tone and approach
    - Key talking points and conversation starters
    - Escalation triggers and approval authorities
    - Success metrics and follow-up requirements

    ---

    ### RISK SCORING METHODOLOGY
    The tool calculates a comprehensive risk score considering:
    - Base churn risk score from customer profile
    - Complaint frequency and recency
    - Interaction sentiment trends  
    - Financial stress indicators
    - Engagement levels and digital adoption
    - Tenure and relationship depth

    ### PRODUCT RECOMMENDATION ENGINE
    Recommendations are generated based on:
    - Current product portfolio gaps
    - Balance thresholds and income levels
    - Spending patterns and channel preferences
    - Life stage and segment characteristics
    - Cross-sell propensity modeling

    ---

    **Parameters:**
    - `customer_id`: Unique customer identifier (required)

    **Returns:**
    Comprehensive JSON response with customer summary card data organized into 
    actionable sections for case worker use.

    **End goal:** Equip case workers with complete customer context to deliver 
    personalized, informed service that drives satisfaction and retention.
    """
    
    config = CUSTOMER_SUMMARY_CONFIG
    logger.debug(f"handle_customer_summary_card: customer_id={customer_id}")
    
    # Extract config values
    customer_db = config['databases']['customer_db']
    account_db = config['databases']['account_db']
    transaction_db = config['databases']['transaction_db']
    complaint_db = config['databases']['complaint_db']
    interaction_db = config['databases']['interaction_db']
    
    customers_table = config['tables']['customers_table']
    accounts_table = config['tables']['accounts_table'] 
    transactions_table = config['tables']['transactions_table']
    complaints_table = config['tables']['complaints_table']
    interactions_table = config['tables']['interactions_table']
    
    recent_days = config['analysis']['recent_transaction_days']
    complaint_lookback = config['analysis']['complaint_lookback_days']
    interaction_lookback = config['analysis']['interaction_lookback_days']
    
    cutoff_date_transactions = datetime.now() - timedelta(days=recent_days)
    cutoff_date_complaints = datetime.now() - timedelta(days=complaint_lookback)
    cutoff_date_interactions = datetime.now() - timedelta(days=interaction_lookback)

    with conn.cursor() as cur:
        # Single optimized query doing calculations in Teradata
        logger.debug("Executing optimized in-database customer summary calculation")
        
        summary_query = f"""
        WITH customer_base AS (
            SELECT c.* FROM {customer_db}.{customers_table} c
            WHERE c.customer_id = ?
        ),
        accounts_agg AS (
            SELECT 
                a.customer_id,
                SUM(CASE WHEN a.account_type IN ('Checking', 'Savings') 
                    THEN a.current_balance ELSE 0 END) AS total_deposits,
                SUM(CASE WHEN a.account_type = 'Credit Card' 
                    THEN ABS(a.current_balance) ELSE 0 END) AS total_credit_used,
                SUM(CASE WHEN a.account_type = 'Credit Card' 
                    THEN a.credit_limit ELSE 0 END) AS total_credit_limit,
                COUNT(CASE WHEN a.account_type = 'Checking' THEN 1 END) AS checking_count,
                COUNT(CASE WHEN a.account_type = 'Savings' THEN 1 END) AS savings_count,
                COUNT(CASE WHEN a.account_type = 'Credit Card' THEN 1 END) AS credit_count
            FROM {account_db}.{accounts_table} a
            WHERE a.customer_id = ?
            GROUP BY a.customer_id
        ),
        transactions_agg AS (
            SELECT 
                t.customer_id,
                COUNT(*) AS txn_count,
                SUM(CASE WHEN t.amount < 0 THEN ABS(t.amount) ELSE 0 END) AS total_spending,
                COUNT(CASE WHEN t.channel IN ('ONLINE', 'ECOMM', 'MOBILE') THEN 1 END) AS digital_txns,
                COUNT(CASE WHEN UPPER(t.description) LIKE '%OVERDRAFT%' THEN 1 END) AS overdraft_count
            FROM {transaction_db}.{transactions_table} t
            WHERE t.customer_id = ? AND t.txn_ts >= ?
            GROUP BY t.customer_id
        ),
        complaints_agg AS (
            SELECT 
                comp.customer_id,
                COUNT(*) AS total_complaints,
                COUNT(CASE WHEN comp.status IN ('Open', 'In Progress') THEN 1 END) AS open_complaints
            FROM {complaint_db}.{complaints_table} comp
            WHERE comp.customer_id = ? AND comp.opened_ts >= ?
            GROUP BY comp.customer_id
        ),
        interactions_agg AS (
            SELECT 
                i.customer_id,
                COUNT(*) AS total_interactions,
                AVG(CASE WHEN i.sentiment_score IS NOT NULL THEN i.sentiment_score ELSE 0 END) AS avg_sentiment
            FROM {interaction_db}.{interactions_table} i
            WHERE i.customer_id = ? AND i.interaction_ts >= ?
            GROUP BY i.customer_id
        )
        SELECT 
            cb.*,
            COALESCE(aa.total_deposits, 0) AS total_deposits,
            COALESCE(aa.total_credit_used, 0) AS total_credit_used,
            COALESCE(aa.total_credit_limit, 0) AS total_credit_limit,
            COALESCE(aa.checking_count, 0) AS checking_count,
            COALESCE(aa.savings_count, 0) AS savings_count,
            COALESCE(aa.credit_count, 0) AS credit_count,
            COALESCE(ta.txn_count, 0) AS txn_count,
            COALESCE(ta.total_spending, 0) AS total_spending,
            COALESCE(ta.digital_txns, 0) AS digital_txns,
            COALESCE(ta.overdraft_count, 0) AS overdraft_count,
            COALESCE(ca.total_complaints, 0) AS total_complaints,
            COALESCE(ca.open_complaints, 0) AS open_complaints,
            COALESCE(ia.total_interactions, 0) AS total_interactions,
            COALESCE(ia.avg_sentiment, 0) AS avg_sentiment,
            -- Simple calculated fields
            (COALESCE(ta.total_spending, 0) / {recent_days}) * 30 AS monthly_spending,
            CASE WHEN ta.txn_count > 0 THEN ta.digital_txns / ta.txn_count ELSE 0 END AS digital_adoption,
            CASE WHEN aa.total_credit_limit > 0 THEN aa.total_credit_used / aa.total_credit_limit ELSE 0 END AS credit_utilization
        FROM customer_base cb
        LEFT JOIN accounts_agg aa ON cb.customer_id = aa.customer_id
        LEFT JOIN transactions_agg ta ON cb.customer_id = ta.customer_id
        LEFT JOIN complaints_agg ca ON cb.customer_id = ca.customer_id
        LEFT JOIN interactions_agg ia ON cb.customer_id = ia.customer_id
        """
        
        cur.execute(summary_query, [
            customer_id,  # customer_base
            customer_id,  # accounts_agg
            customer_id, cutoff_date_transactions.isoformat(),  # transactions_agg
            customer_id, cutoff_date_complaints.isoformat(),    # complaints_agg
            customer_id, cutoff_date_interactions.isoformat()   # interactions_agg
        ])
        
        results = rows_to_json(cur.description, cur.fetchall())
        
        if not results:
            return create_response(
                {"error": f"Customer {customer_id} not found"}, 
                {"tool_name": "customer_summary_card", "customer_id": customer_id}
            )
        
        customer_data = results[0]

        # Get recent transaction details (small result set)
        recent_txns_query = f"""
        SELECT TOP 5 txn_ts, amount, description, channel
        FROM {transaction_db}.{transactions_table}
        WHERE customer_id = ? AND txn_ts >= ?
        ORDER BY txn_ts DESC
        """
        cur.execute(recent_txns_query, [customer_id, cutoff_date_transactions.isoformat()])
        recent_transactions = rows_to_json(cur.description, cur.fetchall())

    # Build summary card using Teradata-calculated metrics
    summary_card = {}
    
    # Convert key fields to appropriate types for comparisons
    churn_risk_score = safe_float(customer_data['churn_risk_score'])
    complaint_propensity_score = safe_float(customer_data['complaint_propensity_score'])
    financial_stress_flag = safe_bool(customer_data['financial_stress_flag'])
    open_complaints = safe_int(customer_data['open_complaints'])
    overdraft_count = safe_int(customer_data['overdraft_count'])
    avg_sentiment = safe_float(customer_data['avg_sentiment'])
    total_deposits = safe_float(customer_data['total_deposits'])
    credit_utilization = safe_float(customer_data['credit_utilization'])
    digital_adoption = safe_float(customer_data['digital_adoption'])
    vip_flag = safe_bool(customer_data['vip_flag'])
    
    # Calculate risk factors using the safely converted data
    risk_factors = []
    primary_risk = "Low"
    
    if churn_risk_score > 0.4:
        risk_factors.append(f"High churn risk ({churn_risk_score:.2f})")
        primary_risk = "Churn"
    if financial_stress_flag:
        risk_factors.append("Financial stress indicators")
        if primary_risk == "Low": primary_risk = "Financial"
    if open_complaints > 0:
        risk_factors.append(f"{open_complaints} open complaints")
        if primary_risk == "Low": primary_risk = "Service"
    if complaint_propensity_score > 0.5:
        risk_factors.append(f"High complaint propensity ({complaint_propensity_score:.2f})")
    if overdraft_count > 0:
        risk_factors.append(f"{overdraft_count} recent overdrafts")
    if avg_sentiment < -0.2:
        risk_factors.append(f"Negative recent sentiment ({avg_sentiment:.2f})")
        if primary_risk == "Low": primary_risk = "Satisfaction"
    
    # Customer Overview
    summary_card['customer_overview'] = {
        'customer_id': customer_data['customer_id'],
        'full_name': customer_data['full_name'],
        'segment': customer_data['segment'],
        'vip_flag': vip_flag,
        'tenure_years': safe_int(customer_data['tenure_years']),
        'annual_income': safe_float(customer_data['annual_income']),
        'ltv_score': safe_float(customer_data['ltv_score']),
        'city_state': f"{customer_data['city']}, {customer_data['state']}"
    }
    
    # Financial Profile (using Teradata aggregations)
    summary_card['financial_profile'] = {
        'total_deposits': total_deposits,
        'total_credit_used': safe_float(customer_data['total_credit_used']),
        'total_credit_limit': safe_float(customer_data['total_credit_limit']),
        'credit_utilization': credit_utilization,
        'account_counts': {
            'checking': safe_int(customer_data['checking_count']),
            'savings': safe_int(customer_data['savings_count']),
            'credit_cards': safe_int(customer_data['credit_count'])
        },
        'monthly_spending': safe_float(customer_data['monthly_spending']),
        'digital_adoption': digital_adoption,
        'financial_stress_flag': financial_stress_flag
    }
    
    # Risk Assessment (using existing scores + calculated metrics)
    summary_card['risk_assessment'] = {
        'primary_risk_area': primary_risk,
        'risk_factors': risk_factors,
        'scores': {
            'churn_risk': churn_risk_score,
            'churn_tier': customer_data['churn_risk_tier'],
            'complaint_propensity': complaint_propensity_score,
            'escalation_risk': safe_float(customer_data['escalation_risk_score']),
            'credit_risk': safe_float(customer_data['credit_risk_score'])
        },
        'indicators': {
            'financial_stress': financial_stress_flag,
            'open_complaints': open_complaints,
            'recent_overdrafts': overdraft_count,
            'avg_sentiment': avg_sentiment
        }
    }
    
    # Recent Activity Summary
    summary_card['recent_activity'] = {
        'transactions': {
            'count': safe_int(customer_data['txn_count']),
            'spending': safe_float(customer_data['total_spending']),
            'monthly_rate': safe_float(customer_data['monthly_spending']),
            'digital_rate': digital_adoption
        },
        'complaints': {
            'total': safe_int(customer_data['total_complaints']),
            'open': open_complaints
        },
        'interactions': {
            'count': safe_int(customer_data['total_interactions']),
            'avg_sentiment': avg_sentiment
        },
        'recent_transactions': recent_transactions
    }
    
    # Product Recommendations (using calculated metrics)
    recommendations = []
    
    if financial_stress_flag or overdraft_count > 0:
        recommendations.append({
            'product': 'Overdraft Protection',
            'priority': 'High',
            'reason': 'Financial stress or recent overdrafts detected'
        })
    
    if total_deposits > 50000 and customer_data['segment'] in ['High Net Worth', 'Emerging Affluent']:
        recommendations.append({
            'product': 'High-Yield Savings',
            'priority': 'High',
            'reason': f'High deposits (${total_deposits:,.0f}) qualify for premium rates'
        })
    
    if credit_utilization > 0.8 and safe_int(customer_data['credit_count']) > 0:
        recommendations.append({
            'product': 'Credit Limit Increase',
            'priority': 'Medium', 
            'reason': f'High utilization ({credit_utilization:.1%}) may impact credit score'
        })
    
    if digital_adoption < 0.3:
        recommendations.append({
            'product': 'Mobile Banking Features',
            'priority': 'Low',
            'reason': f'Low digital adoption ({digital_adoption:.1%})'
        })
    
    summary_card['product_recommendations'] = recommendations
    
    # Case Worker Guidance
    if vip_flag:
        service_level = "VIP white-glove service"
        approach = "Proactive, escalate quickly, personal attention"
    elif len(risk_factors) >= 3:
        service_level = "Critical retention focus"
        approach = "Empathetic, goodwill gestures, supervisor involvement"
    elif len(risk_factors) >= 1:
        service_level = "High-touch service"
        approach = "Address concerns proactively, solution-focused"
    else:
        service_level = "Standard professional"
        approach = "Efficient service, cross-sell opportunities"
    
    talking_points = [
        f"Valued {safe_int(customer_data['tenure_years'])}-year customer",
        f"Total relationship value: ${safe_float(customer_data['ltv_score']):,.0f}"
    ]
    
    if customer_data.get('preferred_channel'):
        talking_points.append(f"Prefers {customer_data['preferred_channel']} channel")
    if primary_risk != "Low":
        talking_points.append(f"Primary risk area: {primary_risk}")
    
    summary_card['case_worker_guidance'] = {
        'service_level': service_level,
        'approach': approach,
        'talking_points': talking_points,
        'risk_factors': risk_factors,
        'escalation_triggers': [
            "VIP customer - escalate quickly" if vip_flag else None,
            f"High {primary_risk.lower()} risk - retention focus" if primary_risk != "Low" else None,
            "Multiple risk factors present" if len(risk_factors) > 2 else None
        ],
        'goodwill_authority': "$500+" if vip_flag else ("$200" if len(risk_factors) > 1 else "$50")
    }
    
    # Clean up None values
    for section in summary_card.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, list):
                    section[key] = [item for item in value if item is not None]
    
    # Metadata
    metadata = {
        "tool_name": "customer_summary_card",
        "customer_id": customer_id,
        "analysis_date": datetime.now().isoformat(),
        "processing_approach": "in_database_aggregation",
        "data_summary": {
            "accounts": safe_int(customer_data['checking_count']) + safe_int(customer_data['savings_count']) + safe_int(customer_data['credit_count']),
            "recent_transactions": safe_int(customer_data['txn_count']),
            "total_complaints": safe_int(customer_data['total_complaints']),
            "recent_interactions": safe_int(customer_data['total_interactions'])
        },
        "risk_factors_count": len(risk_factors),
        "primary_risk_area": primary_risk,
        "recommendations_count": len(recommendations),
        "description": "Optimized customer summary using Teradata in-database processing with safe type conversion"
    }
    
    logger.debug(f"In-database customer summary generated for {customer_id}")
    return create_response(summary_card, metadata)




##########################################################################################################################

def handle_banking_data_query(
    conn: TeradataConnection,
    sql: str | None = None,
    tool_name: str | None = None,
    *args,
    **kwargs
):
    """
    **FLEXIBLE BANKING DATA QUERY TOOL FOR CALL CENTER SUPPORT**

    Execute custom SQL queries against Wells Fargo banking data to answer specific customer 
    service questions that require detailed transaction, account, or operational analysis.

    This tool bridges the gap between pre-built summary tools and raw data access, enabling 
    case workers to get precise answers to customer questions in real-time.

    ### COMMON CALCULATED FIELDS

    #### Total Deposits (from Accounts)
    ```sql
    -- Calculate total deposit balances for customers
    SELECT 
        c.customer_id, c.full_name,
        SUM(CASE WHEN a.account_type IN ('Checking', 'Savings') 
            THEN a.current_balance ELSE 0 END) as total_deposits
    FROM demo_db.wf_customer_profiles c
    JOIN demo_db.wf_accounts a ON c.customer_id = a.customer_id
    WHERE c.vip_flag = 1
    GROUP BY c.customer_id, c.full_name
    HAVING total_deposits > 100000;
    ```

    #### Credit Utilization (from Accounts)
    ```sql
    -- Calculate credit utilization for customers
    SELECT 
        c.customer_id, c.full_name,
        SUM(CASE WHEN a.account_type = 'Credit Card' 
            THEN ABS(a.current_balance) ELSE 0 END) as total_credit_used,
        SUM(CASE WHEN a.account_type = 'Credit Card' 
            THEN a.credit_limit ELSE 0 END) as total_credit_limit,
        CASE WHEN SUM(CASE WHEN a.account_type = 'Credit Card' THEN a.credit_limit ELSE 0 END) > 0
            THEN SUM(CASE WHEN a.account_type = 'Credit Card' THEN ABS(a.current_balance) ELSE 0 END) /
                 SUM(CASE WHEN a.account_type = 'Credit Card' THEN a.credit_limit ELSE 0 END)
            ELSE 0 END as credit_utilization_rate
    FROM demo_db.wf_customer_profiles c
    JOIN demo_db.wf_accounts a ON c.customer_id = a.customer_id
    GROUP BY c.customer_id, c.full_name;
    ```

    ---

    ### WHEN TO USE THIS TOOL

    Use this tool when the other specialized tools don't provide the specific data needed:

    **Transaction Details:**
    - "Show me this customer's last 10 transactions with merchant names"
    - "What ATM withdrawals did this customer make last week?"
    - "Find all transactions over $1000 in the past month"

    **Fee Analysis:**
    - "What fees were charged to this customer in the last 30 days?"
    - "Show all overdraft fees for this customer this year"
    - "List foreign transaction fees from the customer's Europe trip"

    **Account Status & Alerts:**
    - "Are there any holds, blocks, or alerts on this customer's accounts?"
    - "Show pending transactions or authorization holds"
    - "What's the customer's available balance vs current balance?"

    **Agent Performance & Workload:**
    - "Which agents handled the most fraud cases this month?"
    - "Show average resolution time by agent for card replacement issues"
    - "What's the complaint-to-resolution rate for each queue?"

    **Operational Analytics:**
    - "How many customers called about card issues this week?"
    - "What's our average resolution time for fee disputes?"
    - "Show complaint trends by channel over the last month"

    **Historical Research:**
    - "Find all complaints from this customer in the past year"
    - "Show the customer's account opening timeline"
    - "What was this customer's spending pattern before the complaint?"

    ---

    ### AVAILABLE DATA TABLES

    #### Customer Data
    **demo_db.wf_customer_profiles** - Core customer information
    ```sql
    -- Key columns:
    customer_id, first_name, last_name, full_name, full_name_norm, email, phone_primary, 
    address_line1, city, state, zip_code, date_of_birth, customer_since, segment, 
    vip_flag, tenure_years, relationship_manager, preferred_channel, household_id, 
    occupation, annual_income, primary_branch_id, churn_risk_score, churn_risk_tier, 
    churn_model_version, churn_last_scored_date, credit_risk_score, financial_stress_flag,
    complaint_propensity_score, escalation_risk_score, ltv_score, ltv_tier,
    created_date, last_updated
    
    -- NOTE: total_deposits is NOT a column - must calculate from wf_accounts
    ```

    #### Account Data  
    **demo_db.wf_accounts** - Account balances and status
    ```sql
    -- Key columns:
    account_id, customer_id, account_type, account_status, current_balance, 
    available_balance, interest_rate, monthly_fee, credit_limit, opened_date,
    last_transaction_date, overdraft_protection, account_health_score,
    dormancy_risk_score, closure_risk_score, low_engagement_flag, high_maintenance_flag,
    created_date, last_updated
    
    -- Account types: 'Checking', 'Savings', 'Credit Card'
    -- For deposits: SUM current_balance WHERE account_type IN ('Checking', 'Savings')
    -- For credit utilization: current_balance is negative for Credit Card accounts
    ```

    #### Transaction Data
    **demo_db.wf_transactions** - All customer transactions
    ```sql
    -- Key columns:
    transaction_id, account_id, customer_id, txn_ts, posting_date, transaction_type,
    direction, amount, merchant_id, description, channel, status, location,
    fraud_flag, fraud_risk_score, fraud_reason_code, anomaly_score,
    velocity_flag, location_anomaly_flag, amount_anomaly_flag
    ```

    #### Complaint Data
    **demo_db.wf_complaints** - Customer complaints and resolutions
    ```sql
    -- Key columns:
    complaint_id, customer_id, issue_cat_id, opened_ts, closed_ts, status, 
    severity, priority, description_text, agent_notes, resolution_code, 
    refund_amount, sentiment_open, sentiment_close, escalation_risk_score, 
    urgency_score, emotion_primary, origin_transaction_id
    
    -- NOTE: agent_id is NOT in this table - use wf_interactions for agent info
    ```

    #### Interaction Data
    **demo_db.wf_interactions** - Customer service interactions (includes agent info)
    ```sql
    -- Key columns:
    interaction_id, complaint_id, customer_id, account_id, interaction_ts,
    channel, agent_id, summary_text, full_transcript_text, agent_private_notes,
    customer_visible_notes, sentiment_score, disposition, duration_seconds,
    follow_up_required, follow_up_due_ts, priority, satisfaction_predicted,
    resolution_confidence, call_quality_score, key_topics_extracted,
    next_best_action, customer_effort_predicted
    ```

    #### Reference Data
    **demo_db.wf_merchants** - Merchant information for transactions
    ```sql
    -- Key columns:
    merchant_id, merchant_name, mcc, category, city, state, country
    ```

    **demo_db.wf_agents** - Agent information
    ```sql
    -- Key columns:
    agent_id, full_name, queue, experience_level, hire_date, location, active_flag
    ```

    **demo_db.wf_issue_taxonomy** - Issue categorization
    ```sql
    -- Key columns:
    issue_cat_id, level1, level2, level3, active_flag
    -- Hierarchical issue types (level1 > level2 > level3)
    ```

    ---

    ### CORRECTED EXAMPLE QUERIES

    #### Recent Transaction Details
    ```sql
    SELECT TOP 10 
        t.txn_ts, t.amount, t.description, m.merchant_name, t.channel, t.location
    FROM demo_db.wf_transactions t
    LEFT JOIN demo_db.wf_merchants m ON t.merchant_id = m.merchant_id
    WHERE t.customer_id = 'ECN000001'
      AND t.txn_ts >= CURRENT_DATE - 30
    ORDER BY t.txn_ts DESC;
    ```

    #### Fee Analysis
    ```sql
    SELECT t.txn_ts, t.amount, t.description, t.transaction_type
    FROM demo_db.wf_transactions t
    WHERE t.customer_id = 'ECN000001'
      AND t.txn_ts >= CURRENT_DATE - 30
      AND (UPPER(t.description) LIKE '%FEE%' 
           OR UPPER(t.description) LIKE '%OVERDRAFT%'
           OR UPPER(t.description) LIKE '%NSF%')
    ORDER BY t.txn_ts DESC;
    ```

    #### Account Status Check
    ```sql
    SELECT 
        a.account_type, a.account_status, a.current_balance, 
        a.available_balance, a.credit_limit, a.overdraft_protection,
        a.account_health_score, a.last_transaction_date
    FROM demo_db.wf_accounts a
    WHERE a.customer_id = 'ECN000001'
      AND a.account_status = 'Active'
    ORDER BY a.account_type;
    ```

    #### Complaint History with Agent Information (CORRECTED)
    ```sql
    -- To get agent info, must join complaints with interactions
    SELECT DISTINCT
        c.complaint_id, c.opened_ts, c.closed_ts, c.status, c.severity,
        t.level1, t.level2, t.level3, c.resolution_code, c.refund_amount,
        i.agent_id, a.full_name as agent_name, a.queue
    FROM demo_db.wf_complaints c
    LEFT JOIN demo_db.wf_issue_taxonomy t ON c.issue_cat_id = t.issue_cat_id
    LEFT JOIN demo_db.wf_interactions i ON c.complaint_id = i.complaint_id
    LEFT JOIN demo_db.wf_agents a ON i.agent_id = a.agent_id
    WHERE c.customer_id = 'ECN000001'
      AND c.opened_ts >= CURRENT_DATE - 180
    ORDER BY c.opened_ts DESC;
    ```

    #### Agent Performance Analysis (CORRECTED)
    ```sql
    -- Agent performance requires interactions table
    SELECT 
        a.agent_id, a.full_name, a.queue,
        COUNT(DISTINCT i.complaint_id) as complaints_handled,
        AVG(i.satisfaction_predicted) as avg_satisfaction,
        AVG(i.resolution_confidence) as avg_resolution_confidence,
        COUNT(CASE WHEN i.disposition = 'Resolved' THEN 1 END) as resolved_count
    FROM demo_db.wf_agents a
    JOIN demo_db.wf_interactions i ON a.agent_id = i.agent_id
    WHERE i.interaction_ts >= CURRENT_DATE - 30
    GROUP BY a.agent_id, a.full_name, a.queue
    ORDER BY complaints_handled DESC;
    ```

    #### Operational Metrics by Issue Type
    ```sql
    SELECT 
        t.level1 as issue_category, 
        COUNT(*) as total_complaints,
        AVG(CAST(c.closed_ts AS DATE) - CAST(c.opened_ts AS DATE)) as avg_resolution_days,
        AVG(c.sentiment_close - c.sentiment_open) as sentiment_improvement,
        COUNT(CASE WHEN c.refund_amount > 0 THEN 1 END) as refunds_issued
    FROM demo_db.wf_complaints c
    JOIN demo_db.wf_issue_taxonomy t ON c.issue_cat_id = t.issue_cat_id
    WHERE c.opened_ts >= CURRENT_DATE - 30
      AND c.status = 'Resolved'
    GROUP BY t.level1
    ORDER BY total_complaints DESC;
    ```

    #### High-Value Customer Activity (CORRECTED)
    ```sql
    SELECT 
        cp.full_name, cp.segment, cp.ltv_score,
        SUM(CASE WHEN a.account_type IN ('Checking', 'Savings') 
            THEN a.current_balance ELSE 0 END) as total_deposits,
        COUNT(t.transaction_id) as txn_count,
        SUM(CASE WHEN t.amount < 0 THEN ABS(t.amount) ELSE 0 END) as total_spending
    FROM demo_db.wf_customer_profiles cp
    JOIN demo_db.wf_accounts a ON cp.customer_id = a.customer_id
    LEFT JOIN demo_db.wf_transactions t ON cp.customer_id = t.customer_id 
        AND t.txn_ts >= CURRENT_DATE - 30
    WHERE cp.vip_flag = 1
    GROUP BY cp.customer_id, cp.full_name, cp.segment, cp.ltv_score
    HAVING total_deposits > 100000
    ORDER BY total_deposits DESC;
    ```

    #### Complaint-to-Agent Assignment Analysis
    ```sql
    SELECT 
        c.complaint_id, c.customer_id, c.opened_ts, c.status,
        i.agent_id, a.full_name as agent_name, a.queue,
        i.interaction_ts, i.disposition,
        ROW_NUMBER() OVER (PARTITION BY c.complaint_id ORDER BY i.interaction_ts) as interaction_sequence
    FROM demo_db.wf_complaints c
    JOIN demo_db.wf_interactions i ON c.complaint_id = i.complaint_id
    JOIN demo_db.wf_agents a ON i.agent_id = a.agent_id
    WHERE c.opened_ts >= CURRENT_DATE - 7
    ORDER BY c.opened_ts DESC, interaction_sequence;
    ```

    ---

    ### SAFETY GUIDELINES

    #### Row Limits
    - Always include `TOP` clauses for large result sets
    - Recommended limits: Transactions (50), Complaints (20), Customers (100), Interactions (100)

    #### Date Filtering
    - Always filter by date ranges to avoid full table scans
    - Use `ADD_MONTHS(CURRENT_DATE, -n)` for month-based lookbacks
    - Use `CURRENT_DATE - n` for day-based lookbacks (simple integer subtraction)
    - Examples: 
      - Last 90 days: `txn_ts >= CURRENT_DATE - 90`
      - Last 6 months: `txn_ts >= ADD_MONTHS(CURRENT_DATE, -6)`

    #### Important Schema Notes
    - **agent_id** is ONLY in `wf_interactions` table, NOT in `wf_complaints`
    - To get agent info for complaints, JOIN complaints → interactions → agents
    - One complaint can have multiple interactions with different agents
    - Use DISTINCT when joining complaints to interactions to avoid duplicates

    #### Sensitive Data
    - Never query full account numbers, SSNs, or full payment card numbers
    - Use masked fields where available
    - Focus on transaction amounts, dates, and merchant information

    #### Performance
    - Use indexed columns (customer_id, account_id, transaction_id, complaint_id, agent_id) in WHERE clauses
    - Join only necessary tables
    - Aggregate at database level rather than in application code
    
    #### Boolean Values
    - Use 1/0 instead of true/false for boolean comparisons
    - Examples: `vip_flag = 1`, `financial_stress_flag = 0`

    ---

    ### COMMON JOIN PATTERNS

    #### Customer + Accounts + Recent Transactions
    ```sql
    SELECT TOP 20
        c.full_name, a.account_type, a.current_balance, 
        t.txn_ts, t.amount, t.description
    FROM demo_db.wf_customer_profiles c
    JOIN demo_db.wf_accounts a ON c.customer_id = a.customer_id
    JOIN demo_db.wf_transactions t ON a.account_id = t.account_id
    WHERE c.customer_id = 'ECN000001'
    ORDER BY t.txn_ts DESC;
    ```

    #### Complaints + Interactions + Agents (Full Context)
    ```sql
    SELECT 
        c.complaint_id, c.description_text, c.status, c.severity,
        i.interaction_ts, i.channel, i.summary_text, i.disposition,
        a.full_name as agent_name, a.queue, a.experience_level
    FROM demo_db.wf_complaints c
    LEFT JOIN demo_db.wf_interactions i ON c.complaint_id = i.complaint_id
    LEFT JOIN demo_db.wf_agents a ON i.agent_id = a.agent_id
    WHERE c.customer_id = 'ECN000001'
    ORDER BY c.opened_ts DESC, i.interaction_ts ASC;
    ```

    ---

    **Parameters:**
    - `sql`: The SQL query to execute (required)
    - Bind parameters can be passed as keyword arguments

    **Returns:**
    JSON response with query results, column metadata, and execution details.

    **Use this tool when:** You need specific data points that aren't covered by the 
    customer summary, interaction history, or complaint similarity tools. Perfect for 
    answering detailed customer questions or performing operational analysis.
    """

    logger.debug(f"Tool: handle_banking_data_query: Args: sql: {sql}, args={args!r}, kwargs={kwargs!r}")

    if not sql or not sql.strip():
        return create_response(
            {"error": "SQL query is required"}, 
            {"tool_name": "banking_data_query", "error": "No SQL provided"}
        )

    # Basic safety checks
    sql_upper = sql.upper().strip()
    
    # Block dangerous operations
    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return create_response(
                {"error": f"Operation not allowed: {keyword}"}, 
                {"tool_name": "banking_data_query", "error": f"Blocked keyword: {keyword}"}
            )
    
    # Encourage TOP clauses for large tables
    large_tables = ['wf_transactions', 'wf_interactions', 'wf_complaints']
    needs_limit = any(table in sql_upper for table in large_tables)
    if needs_limit and 'TOP' not in sql_upper:
        logger.warning("Query on large table without TOP clause - results may be large")

    try:
        with conn.cursor() as cur:
            # Execute with bind parameters if provided
            if kwargs:
                # Convert keyword args to list for positional binding
                # This is a simple approach - you might need to adjust based on your parameter binding needs
                cur.execute(sql, list(kwargs.values()))
            else:
                cur.execute(sql)
            
            # Fetch results
            raw_rows = cur.fetchall() or []
            data = rows_to_json(cur.description, raw_rows)
            
            # Column metadata
            columns = [
                {
                    "name": col[0],
                    "type": str(col[1]) if col[1] else "unknown"
                }
                for col in (cur.description or [])
            ]

        # Build metadata
        metadata = {
            "tool_name": "banking_data_query",
            "sql": sql,
            "columns": columns,
            "row_count": len(data),
            "execution_time": datetime.now().isoformat(),
            "has_top_clause": 'TOP' in sql_upper,
            "bind_parameters": list(kwargs.keys()) if kwargs else [],
            "tables_accessed": [table for table in ['wf_customer_profiles', 'wf_accounts', 'wf_transactions', 
                                                   'wf_complaints', 'wf_interactions', 'wf_merchants', 
                                                   'wf_agents', 'wf_issue_taxonomy'] if table in sql.lower()],
            "description": "Flexible SQL query execution for detailed banking data analysis"
        }

        # Add warning if result set is large
        if len(data) > 100:
            metadata["warning"] = f"Large result set ({len(data)} rows) - consider adding TOP clause for better performance"

        logger.debug(f"Tool: handle_banking_data_query: metadata: {metadata}")
        return create_response(data, metadata)

    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        error_metadata = {
            "tool_name": "banking_data_query",
            "error": str(e),
            "sql": sql,
            "bind_parameters": list(kwargs.keys()) if kwargs else []
        }
        return create_response(
            {"error": f"Query execution failed: {str(e)}"}, 
            error_metadata
        )