import os
import asyncio
import logging
import signal
import json
import yaml
from typing import Any, List, Optional
from pydantic import Field, BaseModel
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage, TextContent
from dotenv import load_dotenv
import tdfs4ds
import teradataml as tdml
import inspect
from sqlalchemy.engine import Engine, Connection
from typing import Optional


try:
    from teradata_mcp_server import tools as td
except ImportError:
    import tools as td

load_dotenv()

# Load tool configuration from YAML file
with open('configure_tools.yml', 'r') as file:
    config = yaml.safe_load(file)

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(os.path.join("logs", "teradata_mcp_server.log"))],
)
logger = logging.getLogger("teradata_mcp_server")    
logger.info("Starting Teradata MCP server")

# Connect to MCP server
mcp = FastMCP("teradata-mcp-server")

#global shutdown flag
shutdown_in_progress = False

# Initiate connection to Teradata
_tdconn = td.TDConn()

#afm-defect:
_enableEVS = False
# Only attempt to connect to EVS is the system has an EVS installed/configured
if (len(os.getenv("VS_NAME", "").strip()) > 0):
    try:
        _evs    = td.get_evs()
        _enableEVS = True
    except:
        logger.error("Unable to establish connection to EVS, disabling")
        
#afm-defect: moved establish teradataml connection into main TDConn to enable auto-reconnect.
#td.teradataml_connection()



#------------------ Tool utilies  ------------------#
ResponseType = List[types.TextContent | types.ImageContent | types.EmbeddedResource]

def format_text_response(text: Any) -> ResponseType:
    """Format a text response."""
    if isinstance(text, str):
        try:
            # Try to parse as JSON if it's a string
            parsed = json.loads(text)
            return [types.TextContent(
                type="text", 
                text=json.dumps(parsed, indent=2, ensure_ascii=False)
            )]
        except json.JSONDecodeError:
            # If not JSON, return as plain text
            return [types.TextContent(type="text", text=str(text))]
    # For non-string types, convert to string
    return [types.TextContent(type="text", text=str(text))]

def format_error_response(error: str) -> ResponseType:
    """Format an error response."""
    return format_text_response(f"Error: {error}")

def execute_db_tool(tool, *args, **kwargs):
    """
    Execute a database tool with the given connection and arguments.
    Currently support both tools expecting DB API or SQLAlchemy engine:
      - If annotated Connection, pass SQLAlchemy engine
      - Otherwise, pass raw DB-API connection
    The second option should be eventually retired as all tools move to SQLAlchemy.
    """
    global _tdconn
    # (Re)initialize if needed
    if not getattr(_tdconn, "engine", None):
        logger.info("Reinitializing TDConn")
        _tdconn = td.TDConn()

    # Check is the first argument of the tool is a SQLAlchemy Connection
    sig = inspect.signature(tool)
    first_param = next(iter(sig.parameters.values()))
    ann = first_param.annotation
    use_sqla = inspect.isclass(ann) and issubclass(ann, Connection)

    try:
        if use_sqla:
            # Use a Connection that has .execute()
            with _tdconn.engine.connect() as conn:
                result = tool(conn, *args, **kwargs)
        else:
            # Raw DB-API path
            raw = _tdconn.engine.raw_connection()
            try:
                result = tool(raw, *args, **kwargs)
            finally:
                raw.close()

        return format_text_response(result)

    except Exception as e:
        logger.error(f"Error in execute_db_tool: {e}", exc_info=True)
        return format_error_response(str(e))
    

def execute_vs_tool(tool, *args, **kwargs) -> ResponseType:
    global _evs
    global _enableEVS

    if _enableEVS:
        try:
            return format_text_response(tool(_evs, *args, **kwargs))
        except Exception as e:
            if "401" in str(e) or "Session expired" in str(e):
                logger.warning("EVS session expired, refreshing …")
                _evs = td.evs_connect.refresh_evs()
                try:
                    return format_text_response(tool(_evs, *args, **kwargs))
                except Exception as retry_err:
                    logger.error(f"EVS retry failed: {retry_err}")
                    return format_error_response(f"After refresh, still failed: {retry_err}")
    
            logger.error(f"EVS tool error: {e}")
            return format_error_response(str(e))
    else:
        return format_error_response("Enterprise Vector Store is not available on this server.")

    
#------------------ Base Tools  ------------------#

if config['base']['allmodule']:
    if config['base']['tool']['base_readQuery']:
        @mcp.tool(description="Executes a SQL query to read from the database.")
        async def base_readQuery(
            sql: str = Field(description="SQL that reads from the database to run", default=""),
            ) -> ResponseType:
            """Executes a SQL query to read from the database."""
            return execute_db_tool( td.handle_base_readQuery, sql=sql) 

    if config['base']['tool']['base_writeQuery']:
        @mcp.tool(description="Executes a SQL query to write to the database.")
        async def base_writeQuery(
            sql: str = Field(description="SQL that writes to the database to run", default=""),
            ) -> ResponseType:
            """Executes a SQL query to write to the database."""
            return execute_db_tool( td.handle_base_writeQuery, sql=sql) 

    if config['base']['tool']['base_tableDDL']:
        @mcp.tool(description="Display table DDL definition.")
        async def base_tableDDL(
            db_name: str = Field(description="Database name", default=""),
            table_name: str = Field(description="table name", default=""),
            ) -> ResponseType:
            """Display table DDL definition."""
            return execute_db_tool( td.handle_base_tableDDL, db_name=db_name, table_name=table_name)    

    if config['base']['tool']['base_databaseList']:
        @mcp.tool(description="List all databases in the Teradata System.")
        async def base_databaseList() -> ResponseType:
            """List all databases in the Teradata System."""
            return execute_db_tool( td.handle_base_databaseList)

    if config['base']['tool']['base_tableList']:
        @mcp.tool(description="List objects in a database.")
        async def base_tableList(
            db_name: str = Field(description="database name", default=""),
            ) -> ResponseType:
            """List objects in a database."""
            return execute_db_tool( td.handle_base_tableList, db_name=db_name)

    if config['base']['tool']['base_columnDescription']:
        @mcp.tool(description="Show detailed column information about a database table.")
        async def base_columnDescription(
            db_name: str = Field(description="Database name", default=""),
            obj_name: str = Field(description="table name", default=""),
            ) -> ResponseType:
            """Show detailed column information about a database table."""
            return execute_db_tool( td.handle_base_columnDescription, db_name=db_name, obj_name=obj_name)

    if config['base']['tool']['base_tablePreview']:
        @mcp.tool(description="Get data samples and structure overview from a database table.")
        async def base_tablePreview(
            db_name: str = Field(description="Database name", default=""),
            table_name: str = Field(description="table name", default=""),
            ) -> ResponseType:
            """Get data samples and structure overview from a database table."""
            return execute_db_tool( td.handle_base_tablePreview, table_name=table_name, db_name=db_name)

    if config['base']['tool']['base_tableAffinity']:
        @mcp.tool(description="Get tables commonly used together by database users, this is helpful to infer relationships between tables.")
        async def base_tableAffinity(
            db_name: str = Field(description="Database name", default=""),
            obj_name: str = Field(description="Table or view name", default=""),
            ) -> ResponseType:
            """Get tables commonly used together by database users, this is helpful to infer relationships between tables."""
            return execute_db_tool( td.handle_base_tableAffinity, obj_name=obj_name, db_name=db_name)

    if config['base']['tool']['base_tableUsage']:
        @mcp.tool(description="Measure the usage of a table and views by users in a given schema, this is helpful to infer what database objects are most actively used or drive most value.")
        async def base_tableUsage(
            db_name: str = Field(description="Database name", default=""),
            ) -> ResponseType:
            """Measure the usage of a table and views by users in a given schema, this is helpful to infer what database objects are most actively used or drive most value."""
            return execute_db_tool( td.handle_base_tableUsage, db_name=db_name)

    if config['base']['prompt']['base_query']:
        @mcp.prompt()
        async def base_query(qry: str) -> UserMessage:
            """Create a SQL query against the database"""
            return UserMessage(role="user", content=TextContent(type="text", text=td.handle_base_query.format(qry=qry)))

    if config['base']['prompt']['base_tableBusinessDesc']:
        @mcp.prompt()
        async def base_tableBusinessDesc(database_name: str, table_name: str) -> UserMessage:
            """Create a business description of the table and columns."""
            return UserMessage(role="user", content=TextContent(type="text", text=td.handle_base_tableBusinessDesc.format(database_name=database_name, table_name=table_name)))

    if config['base']['prompt']['base_databaseBusinessDesc']:
        @mcp.prompt()
        async def base_databaseBusinessDesc(database_name: str) -> UserMessage:
            """Create a business description of the database."""
            return UserMessage(role="user", content=TextContent(type="text", text=td.handle_base_databaseBusinessDesc.format(database_name=database_name)))

#------------------ DBA Tools  ------------------#
if config['dba']['allmodule']:
    if config['dba']['tool']['dba_userSqlList']:
        @mcp.tool(description="Get a list of SQL run by a user in the last number of days if a user name is provided, otherwise get list of all SQL in the last number of days.")
        async def dba_userSqlList(
            user_name: str = Field(description="user name", default=""),
            no_days: int = Field(description="number of days to look back", default=7),
            ) -> ResponseType:
            """Get a list of SQL run by a user."""
            return execute_db_tool( td.handle_dba_userSqlList, user_name=user_name, no_days=no_days)

    if config['dba']['tool']['dba_tableSqlList']:
        @mcp.tool(description="Get a list of SQL run against a table in the last number of days ")
        async def dba_tableSqlList(
            table_name: str = Field(description="table name", default=""),
            no_days: int = Field(description="number of days to look back", default=7),
            ) -> ResponseType:
            """Get a list of SQL run by a user."""
            return execute_db_tool( td.handle_dba_tableSqlList, table_name=table_name, no_days=no_days)

    if config['dba']['tool']['dba_tableSpace']:
        @mcp.tool(description="Get table space used for a table if table name is provided or get table space for all tables in a database if a database name is provided.")
        async def dba_tableSpace(
            db_name: str = Field(description="Database name", default=""),
            table_name: str = Field(description="table name", default=""),
            ) -> ResponseType:
            """Get table space used for a table if table name is provided or get table space for all tables in a database if a database name is provided."""
            return execute_db_tool( td.handle_dba_tableSpace, db_name=db_name, table_name=table_name)

    if config['dba']['tool']['dba_databaseSpace']:
        @mcp.tool(description="Get database space if database name is provided, otherwise get all databases space allocations.")
        async def dba_databaseSpace(
            db_name: str = Field(description="Database name", default=""),
            ) -> ResponseType:
            """Get database space if database name is provided, otherwise get all databases space allocations."""
            return execute_db_tool( td.handle_dba_databaseSpace, db_name=db_name)

    if config['dba']['tool']['dba_databaseVersion']:
        @mcp.tool(description="Get Teradata database version information.")
        async def dba_databaseVersion() -> ResponseType:
            """Get Teradata database version information."""
            return execute_db_tool( td.handle_dba_databaseVersion)

    if config['dba']['tool']['dba_resusageSummary']:
        @mcp.tool(description="Get the Teradata system usage summary metrics by weekday and hour for each workload type and query complexity bucket.")
        async def dba_resusageSummary() -> ResponseType:
            """Get the Teradata system usage summary metrics by weekday and hour."""
            return execute_db_tool( td.handle_dba_resusageSummary, dimensions=["hourOfDay", "dayOfWeek"])

    if config['dba']['tool']['dba_resusageUserSummary']:
        @mcp.tool(description="Get the Teradata system usage summary metrics by user on a specified date, or day of week and hour of day.")
        async def dba_resusageUserSummary(
            user_name: str = Field(description="Database user name", default=""),
            date: str = Field(description="Date to analyze, formatted as `YYYY-MM-DD`", default=""),
            dayOfWeek: str = Field(description="Day of week to analyze", default=""),
            hourOfDay: str = Field(description="Hour of day to analyze", default=""),
            ) -> ResponseType:
            """Get the Teradata system usage summary metrics by user on a specified date, or day of week and hour of day."""
            return execute_db_tool( td.handle_dba_resusageSummary, dimensions=["UserName", "hourOfDay", "dayOfWeek"], user_name=user_name, date=date, dayOfWeek=dayOfWeek, hourOfDay=hourOfDay)

    if config['dba']['tool']['dba_flowControl']:
        @mcp.tool(description="Get the Teradata flow control metrics.")
        async def dba_flowControl() -> ResponseType:
            """Get the Teradata flow control metrics."""
            return execute_db_tool( td.handle_dba_flowControl)

    if config['dba']['tool']['dba_featureUsage']:
        @mcp.tool(description="Get the user feature usage metrics.")
        async def dba_featureUsage() -> ResponseType:
            """Get the user feature usage metrics.""" 
            return execute_db_tool( td.handle_dba_featureUsage)

    if config['dba']['tool']['dba_userDelay']:
        @mcp.tool(description="Get the Teradata user delay metrics.")
        async def dba_userDelay() -> ResponseType:
            """Get the Teradata user delay metrics."""
            return execute_db_tool( td.handle_dba_userDelay)

    if config['dba']['tool']['dba_tableUsageImpact']:
        @mcp.tool(description="Measure the usage of a table and views by users, this is helpful to understand what user and tables are driving most resource usage at any point in time.")
        async def dba_tableUsageImpact(
            db_name: str = Field(description="Database name", default=""),
            user_name: str = Field(description="User name", default=""),    
            ) -> ResponseType:
            """Measure the usage of a table and views by users, this is helpful to understand what user and tables are driving most resource usage at any point in time."""
            return execute_db_tool( td.handle_dba_tableUsageImpact, db_name=db_name,  user_name=user_name)

    if config['dba']['tool']['dba_sessionInfo']:
        @mcp.tool(description="Get the Teradata session information for user.")
        async def dba_sessionInfo(
            user_name: str = Field(description="User name", default=""),
            ) -> ResponseType:
            """Get the Teradata session information for user."""
            return execute_db_tool( td.handle_dba_sessionInfo, user_name=user_name)

    if config['dba']['prompt']['dba_databaseHealthAssessment']:
        @mcp.prompt()
        async def dba_databaseHealthAssessment() -> UserMessage:
            """Create a database health assessment for a Teradata system."""
            return UserMessage(role="user", content=TextContent(type="text", text=td.handle_dba_databaseHealthAssessment))

    if config['dba']['prompt']['dba_userActivityAnalysis']:
        @mcp.prompt()
        async def dba_userActivityAnalysis() -> UserMessage:
            """Create a user activity analysis for a Teradata system."""
            return UserMessage(role="user", content=TextContent(type="text", text=td.handle_dba_userActivityAnalysis))

    if config['dba']['prompt']['dba_tableArchive']:
        @mcp.prompt()
        async def dba_tableArchive() -> UserMessage:
            """Create a table archive strategy for database tables."""
            return UserMessage(role="user", content=TextContent(type="text", text=td.handle_dba_tableArchive))

    if config['dba']['prompt']['dba_databaseLineage']:
        @mcp.prompt()
        async def dba_databaseLineage(database_name: str, number_days: int) -> UserMessage:
            """Create a database lineage map for tables in a database."""
            return UserMessage(role="user", content=TextContent(type="text", text=td.handle_dba_databaseLineage.format(database_name=database_name, number_days=number_days)))

    if config['dba']['prompt']['dba_tableDropImpact']:
        @mcp.prompt()
        async def dba_tableDropImpact(database_name: str, table_name: str, number_days: int) -> UserMessage:
            """Assess the impact of dropping a table."""
            return UserMessage(role="user", content=TextContent(type="text", text=td.handle_dba_tableDropImpact.format(database_name=database_name, table_name=table_name, number_days=number_days)))

#------------------ Data Quality Tools  ------------------#

if config['qlty']['allmodule']:
    if config['qlty']['tool']['qlty_missingValues']:
        @mcp.tool(description="Get the column names that having missing values in a table.")
        async def qlty_missingValues(
            table_name: str = Field(description="table name", default=""),
            ) -> ResponseType:
            """Get the column names that having missing values in a table."""
            return execute_db_tool( td.handle_qlty_missingValues, table_name=table_name)

    if config['qlty']['tool']['qlty_negativeValues']:
        @mcp.tool(description="Get the column names that having negative values in a table.")
        async def qlty_negativeValues(
            table_name: str = Field(description="table name", default=""),
            ) -> ResponseType:
            """Get the column names that having negative values in a table."""
            return execute_db_tool( td.handle_qlty_negativeValues, table_name=table_name)

    if config['qlty']['tool']['qlty_distinctCategories']:
        @mcp.tool(description="Get the destinct categories from column in a table.")
        async def qlty_distinctCategories(
            table_name: str = Field(description="table name", default=""),
            col_name: str = Field(description="column name", default=""),
            ) -> ResponseType:
            """Get the destinct categories from column in a table."""
            return execute_db_tool( td.handle_qlty_distinctCategories, table_name=table_name, col_name=col_name)

    if config['qlty']['tool']['qlty_standardDeviation']:
        @mcp.tool(description="Get the standard deviation from column in a table.")
        async def qlty_standardDeviation(
            table_name: str = Field(description="table name", default=""),
            col_name: str = Field(description="column name", default=""),
            ) -> ResponseType:
            """Get the standard deviation from column in a table."""
            return execute_db_tool( td.handle_qlty_standardDeviation, table_name=table_name, col_name=col_name)

    if config['qlty']['tool']['qlty_columnSummary']:
        @mcp.tool(description="Get the column summary statistics for a table.")
        async def qlty_columnSummary(
            table_name: str = Field(description="table name", default=""),
            ) -> ResponseType:
            """Get the column summary statistics for a table."""
            return execute_db_tool( td.handle_qlty_columnSummary, table_name=table_name)

    if config['qlty']['tool']['qlty_univariateStatistics']:
        @mcp.tool(description="Get the univariate statistics for a table.")
        async def qlty_univariateStatistics(
            table_name: str = Field(description="table name", default=""),
            col_name: str = Field(description="column name", default=""),
            ) -> ResponseType:
            """Get the univariate statistics for a table."""
            return execute_db_tool( td.handle_qlty_univariateStatistics, table_name=table_name, col_name=col_name)

    if config['qlty']['tool']['qlty_rowsWithMissingValues']:
        @mcp.tool(description="Get the rows with missing values in a table.")
        async def qlty_rowsWithMissingValues(
            table_name: str = Field(description="table name", default=""),
            col_name: str = Field(description="column name", default=""),
            ) -> ResponseType:
            """Get the rows with missing values in a table."""
            return execute_db_tool( td.handle_qlty_rowsWithMissingValues, table_name=table_name, col_name=col_name)

    if config['qlty']['prompt']['qlty_databaseQuality']:
        @mcp.prompt()
        async def qlty_databaseQuality(database_name: str) -> UserMessage:
            """Assess the data quality of a database."""
            return UserMessage(role="user", content=TextContent(type="text", text=td.handle_qlty_databaseQuality.format(database_name=database_name)))

# ------------------ RAG Tools ------------------ #

if config['rag']['allmodule']:
    if config['rag']['tool']['rag_executeWorkflow']:
        
        # Get the RAG version from rag_config.yaml instead of configure_tools.yml
        try:
            with open('rag_config.yaml', 'r') as file:  # ✅ CORRECT - same pattern as configure_tools.yml
                rag_config = yaml.safe_load(file)
            rag_version = rag_config.get('version', 'byom')
            default_k = rag_config.get('retrieval', {}).get('default_k', 10)  # Get default_k from config
            logger.info(f"RAG config loaded: version={rag_version}, default_k={default_k}")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load rag_config.yaml: {e}. Using default version 'byom'")
            rag_version = 'byom'
            default_k = 10
        
        if rag_version == 'byom':
            @mcp.tool(description="""
            Execute complete RAG workflow to answer user questions based on document context.
            This tool handles the entire RAG pipeline in a single step when a user query is tagged with /rag.

            WORKFLOW STEPS (executed automatically using ONNXEmbeddings):
            1. Configuration setup using configurable values from rag_config.yaml
            2. Store user query with '/rag ' prefix stripping  
            3. Generate query embeddings (tokenization + embedding using mldb.ONNXEmbeddings)
            4. Perform semantic search against precomputed chunk embeddings
            5. Return context chunks for answer generation

            CONFIGURATION VALUES (from rag_config.yaml):
            - All database names, table names, and model settings are configurable
            - Vector store metadata fields are dynamically detected
            - Embedding parameters are configurable
            - Default chunk retrieval count is configurable
            - Default values are provided as fallback

            TECHNICAL DETAILS:
            - Strips the '/rag ' prefix if present from user questions
            - Creates query table if it does not exist (columns: id, txt, created_ts)
            - Retrieves the most recent user question from the configured table
            - Uses the tokenizer and embedding model selected via model_id configuration
            - Runs tokenization and embedding using mldb.ONNXEmbeddings UDF
            - Stores resulting embeddings and metadata in the configured output table
            - Uses cosine similarity via TD_VECTORDISTANCE to compare embedded query against precomputed chunk embeddings
            - Returns the top-k matching chunks from the configured vector store
            - Each result includes chunk text, similarity score, and any metadata fields specified in config

            CRITICAL ANSWERING RULES:
            - Answer ONLY using retrieved chunks - no external knowledge, speculation, or inference
            - Quote source content directly without paraphrasing, summarizing, or rewriting
            - If no relevant context found: "Not enough information found in the provided context"
            - If partial context: "The available context does not fully answer the question"
            - Include document/page references when available (e.g., "On page 2 of 'demo_policy.pdf'...")
            - Execute entire workflow silently without showing function calls to user - only show final answer

            LANGUAGE RESTRICTIONS:
            - Do not say "According to the context" or "The context says" - just quote directly
            - Do not say "It can be inferred that..." - no inference allowed
            - Use exact or near-verbatim quotes only
            - No paraphrasing, summarizing, or adding transitions between quotes
            - Copy content precisely - no expansion or interpretation

            BEHAVIORAL REQUIREMENTS:
            - Execute all RAG workflow steps silently without showing tool calls to users
            - Only display final answers based strictly on retrieved context chunks
            - Do not use external knowledge or speculation - only quote retrieved content directly
            - RAG mode only triggers with explicit /rag prefix - do not enter RAG mode implicitly
            - Hide all tool execution steps from user
            - Provide clean, direct response with only the answer based on retrieved chunks
            - This is a complete end-to-end RAG workflow - no additional tools needed

            EXECUTION: Run completely silently - user only sees their question and the final context-grounded answer.
            """)
            async def rag_executeWorkflow(
                question: str = Field(..., description="User's natural language question, optionally prefixed with '/rag '"),
                k: int = Field(None, description=f"Number of top matching chunks to retrieve for context (uses config default of {default_k} if not specified)"),
            ) -> ResponseType:
                return execute_db_tool(td.handle_rag_executeWorkflow, question=question, k=k)
                
        elif rag_version == 'ivsm':
            @mcp.tool(description="""
            Execute complete RAG workflow to answer user questions based on document context.
            This tool handles the entire RAG pipeline in a single step when a user query is tagged with /rag.

            WORKFLOW STEPS (executed automatically using IVSM functions):
            1. Configuration setup using configurable values from rag_config.yaml
            2. Store user query with '/rag ' prefix stripping  
            3. Tokenize query using ivsm.tokenizer_encode
            4. Create embedding view using ivsm.IVSM_score
            5. Convert embeddings to vector columns using ivsm.vector_to_columns
            6. Perform semantic search against precomputed chunk embeddings

            CONFIGURATION VALUES (from rag_config.yaml):
            - All database names, table names, and model settings are configurable
            - Vector store metadata fields are dynamically detected
            - Embedding parameters are configurable
            - Default chunk retrieval count is configurable
            - Default values are provided as fallback

            TECHNICAL DETAILS:
            - Strips the '/rag ' prefix if present from user questions
            - Creates query table if it does not exist (columns: id, txt, created_ts)
            - Selects the most recent row from the query table, runs it through the ONNX tokenizer
            - Creates temporary view v_topics_tokenized containing 'id', 'txt', 'input_ids', and 'attention_mask'
            - Reads from the tokenized view and applies the ONNX model to create embeddings
            - Creates or replaces embedding view v_topics_embeddings which includes the original input and sentence_embedding column
            - Converts the sentence embedding from the embedding view into vector columns for similarity search
            - Creates or replaces a physical table to store the latest query embeddings for use in similarity search
            - Uses cosine similarity via TD_VECTORDISTANCE to compare embedded query against precomputed chunk embeddings
            - Returns the top-k matching chunks from the configured vector store
            - Each result includes chunk text, similarity score, and any metadata fields specified in config

            CRITICAL ANSWERING RULES:
            - Answer ONLY using retrieved chunks - no external knowledge, speculation, or inference
            - Quote source content directly without paraphrasing, summarizing, or rewriting
            - If no relevant context found: "Not enough information found in the provided context"
            - If partial context: "The available context does not fully answer the question"
            - Include document/page references when available (e.g., "On page 2 of 'demo_policy.pdf'...")
            - Execute entire workflow silently without showing function calls to user - only show final answer

            LANGUAGE RESTRICTIONS:
            - Do not say "According to the context" or "The context says" - just quote directly
            - Do not say "It can be inferred that..." - no inference allowed
            - Use exact or near-verbatim quotes only
            - No paraphrasing, summarizing, or adding transitions between quotes
            - Copy content precisely - no expansion or interpretation

            BEHAVIORAL REQUIREMENTS:
            - Execute all RAG workflow steps silently without showing tool calls to users
            - Only display final answers based strictly on retrieved context chunks
            - Do not use external knowledge or speculation - only quote retrieved content directly
            - RAG mode only triggers with explicit /rag prefix - do not enter RAG mode implicitly
            - Hide all tool execution steps from user
            - Provide clean, direct response with only the answer based on retrieved chunks
            - This is a complete end-to-end RAG workflow using IVSM functions - no additional tools needed

            EXECUTION: Run completely silently - user only sees their question and the final context-grounded answer.
            """)
            async def rag_executeWorkflow(
                question: str = Field(..., description="User's natural language question, optionally prefixed with '/rag '"),
                k: int = Field(None, description=f"Number of top matching chunks to retrieve for context (uses config default of {default_k} if not specified)"),
            ) -> ResponseType:
                return execute_db_tool(td.handle_rag_executeWorkflow_ivsm, question=question, k=k)
        
        else:
            raise ValueError(f"Invalid RAG version: {rag_version}. Must be 'byom' or 'ivsm'")

    if config['rag']['prompt']['rag_guidelines']:
        @mcp.prompt()
        async def rag_guidelines() -> UserMessage:
            return UserMessage(
                role="user",
                content=TextContent(type="text", text=td.rag_guidelines)
            )

if config.get('customer', {}).get('allmodule', False):
    if config['customer']['tool']['customer_meetingPrep']:
        @mcp.tool(description="""
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
        """)
        async def customer_meetingPrep(
            customer_name: str = Field(..., description="Name of the customer for analysis (e.g., 'TechFlow Solutions', 'Global Manufacturing Corp'). Supports partial matching. REQUIRED - if not provided in user question, ask user to specify which customer."),
            meeting_type: str = Field("general", description="Type of meeting: general, renewal, expansion, support, strategic, quarterly_review"),
            lookback_months: int = Field(6, description="Number of months of conversation history to analyze (1-12 months)"),
        ) -> ResponseType:
            """Consolidate customer conversation data for AI-powered meeting preparation recommendations."""
            return execute_db_tool(
                td.handle_customer_meetingPrep,
                customer_name=customer_name,
                meeting_type=meeting_type,
                lookback_months=lookback_months
            )
        
#------------------ Financial RAG Tools  ------------------#

if config.get('financial_reports', {}).get('allmodule', False):
    if config['financial_reports']['tool'].get('financial_rag_analysis', False):
        
        # Load financial RAG config to get defaults
        try:
            with open('financial_rag_config.yaml', 'r') as file:
                financial_config = yaml.safe_load(file)
            default_k_per_year = financial_config.get('retrieval', {}).get('default_k_per_year', 5)
            default_k_global = financial_config.get('retrieval', {}).get('default_k_global', 20)
            logger.info(f"Financial RAG config loaded: default_k_per_year={default_k_per_year}, default_k_global={default_k_global}")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load financial_rag_config.yaml: {e}. Using defaults")
            default_k_per_year = 5
            default_k_global = 20

        @mcp.tool(description=f"""
        Execute financial reports RAG analysis for ICICI Bank annual reports with intelligent multi-year retrieval.
        
        This tool performs sophisticated financial analysis by retrieving relevant chunks from ICICI Bank annual reports
        across multiple years (2011-2015) based on LLM-parsed parameters from the user's question.

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
        - EXTRACT YEARS: Parse years from user queries (e.g., "2011 to 2015" → [2011,2012,2013,2014,2015])
        - AVAILABLE YEARS: 2011, 2012, 2013, 2014, 2015 (ICICI Bank annual reports)
        - DETERMINE ANALYSIS TYPE:
          * "temporal": For trend analysis, growth patterns, evolution over time
          * "comparative": For side-by-side comparisons between years or periods  
          * "general": For single-point factual questions or definitions
        - SET K: Leave as None for smart defaults, or specify total chunks needed

        EXAMPLE PARAMETER EXTRACTION:
        - "How did ICICI's revenue grow from 2011 to 2015?" → years=[2011,2012,2013,2014, 2015], analysis_type="temporal"
        - "Compare ICICI's loan portfolio in 2011 vs 2015" → years=[2011,2015], analysis_type="comparative"  
        - "What was ICICI's main business in 2011?" → years=[2011], analysis_type="general"
        - "ICICI's risk management strategy" → years=None, analysis_type="general"

        RETRIEVED CONTEXT INCLUDES:
        - Clean chunk text optimized for financial analysis (no bloat metadata)
        - Source document names for citations (e.g., "ICICI_Bank_Annual_Report_FY2011.pdf")
        - Report year information for temporal context (report_year field)
        - Section titles for content context (e.g., "Financial Performance", "Risk Management")
        - Similarity scores for relevance assessment
        - Chunk position numbers for precise citations

        DATA SCHEMA (Clean, no bloat):
        - txt: Financial report content
        - doc_name: Source PDF filename
        - report_year: Year of the report (2011-2015)
        - section_title: Report section name
        - chunk_num: Position within document
        - similarity: Relevance score

        CRITICAL ANSWERING RULES:
        - Answer ONLY using retrieved chunks - no external knowledge about ICICI Bank
        - Quote source content directly without paraphrasing or summarizing
        - Include year and document references for citations
        - For multi-year queries, organize analysis chronologically
        - For comparative queries, provide balanced analysis of compared periods
        - If insufficient context: "Not enough information found in the provided annual reports"

        TEMPORAL ANALYSIS GUIDELINES:
        - Show progression over time for temporal queries
        - Highlight year-over-year changes and trends
        - Compare metrics across the requested time period
        - Use retrieved chunks to support trend observations

        EXECUTION: Run completely silently - user only sees the final financial analysis based on retrieved context.
        """)
        async def financial_rag_analysis(
            question: str = Field(..., description="User's financial analysis question about ICICI Bank"),
            years: Optional[List[int]] = Field(None, description="List of specific years to analyze (e.g., [2011,2012,2013] for multi-year or [2012] for single year). Available years: 2011-2015. LLM should extract from user query."),
            analysis_type: str = Field("general", description="Type of analysis: 'temporal' (trends over time), 'comparative' (compare periods), or 'general' (standard queries)"),
            k: int = Field(None, description=f"Total number of chunks to retrieve. Uses smart defaults: {default_k_per_year} per year for multi-year queries, {default_k_global} for single-year queries"),
        ) -> ResponseType:
            return execute_db_tool(td.handle_financial_rag_analysis, question=question, years=years, analysis_type=analysis_type, k=k)

#------------------ SQL Clustering Tools  ------------------#

if config['sql_clustering']['allmodule']:
    if config['sql_clustering']['tool']['sql_clustering_executeFullPipeline']:
        
        # Get the SQL clustering version and default parameters from config
        try:
            with open('sql_clustering_config.yaml', 'r') as file:
                sql_clustering_config = yaml.safe_load(file)
            clustering_version = sql_clustering_config.get('version', 'ivsm')
            default_optimal_k = sql_clustering_config.get('clustering', {}).get('optimal_k', 14)
            default_max_queries = sql_clustering_config.get('clustering', {}).get('max_queries', 10000)
            logger.info(f"SQL clustering config loaded: version={clustering_version}, optimal_k={default_optimal_k}")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load sql_clustering_config.yaml: {e}. Using defaults")
            clustering_version = 'ivsm'
            default_optimal_k = 14
            default_max_queries = 10000

        @mcp.tool(description=f"""
        **COMPLETE SQL QUERY CLUSTERING PIPELINE FOR HIGH-USAGE QUERY OPTIMIZATION**

        This tool executes the entire SQL query clustering workflow to identify and analyze high CPU usage queries for optimization opportunities. It's designed for database performance analysts and DBAs who need to systematically identify query optimization candidates.

        **FULL PIPELINE WORKFLOW:**
        1. **Query Log Extraction**: Extracts SQL queries from DBC.DBQLSqlTbl with comprehensive performance metrics
        2. **Performance Metrics Calculation**: Computes CPU skew, I/O skew, PJI (Physical to Logical I/O ratio), UII (Unit I/O Intensity)
        3. **Query Tokenization**: Tokenizes SQL text using {sql_clustering_config.get('model', {}).get('model_id', 'bge-small-en-v1.5')} tokenizer via ivsm.tokenizer_encode
        4. **Embedding Generation**: Creates semantic embeddings using ivsm.IVSM_score with ONNX models
        5. **Vector Store Creation**: Converts embeddings to vector columns via ivsm.vector_to_columns
        6. **K-Means Clustering**: Groups similar queries using TD_KMeans with optimal K from configuration
        7. **Silhouette Analysis**: Calculates clustering quality scores using TD_Silhouette
        8. **Statistics Generation**: Creates comprehensive cluster statistics with performance aggregations

        **PERFORMANCE METRICS EXPLAINED:**
        - **AMPCPUTIME**: Total CPU seconds across all AMPs (primary optimization target)
        - **CPUSKW/IOSKW**: CPU/I/O skew ratios (>2.0 indicates distribution problems)
        - **PJI**: Physical-to-Logical I/O ratio (higher = more CPU-intensive)
        - **UII**: Unit I/O Intensity (higher = more I/O-intensive relative to CPU)
        - **LogicalIO**: Total logical I/O operations (indicates scan intensity)
        - **NumSteps**: Query plan complexity (higher = more complex plans)

        **CONFIGURATION (from sql_clustering_config.yaml):**
        - Uses top {default_max_queries} queries by CPU time (configurable)
        - Creates {default_optimal_k} clusters by default (configurable via optimal_k parameter)
        - Embedding model: {sql_clustering_config.get('model', {}).get('model_id', 'bge-small-en-v1.5')}
        - Vector dimensions: {sql_clustering_config.get('embedding', {}).get('vector_length', 384)}
        - All database and table names are configurable

        **TABLES CREATED:**
        - feature_ext_db.sql_query_log_main (raw query log with metrics)
        - feature_ext_db.sql_log_tokenized_for_embeddings (tokenized queries)
        - feature_ext_db.sql_log_embeddings (raw embeddings)
        - feature_ext_db.sql_log_embeddings_store (vector store format)
        - feature_ext_db.sql_query_clusters (final clustered queries)
        - feature_ext_db.query_cluster_stats (cluster performance statistics)

        **OPTIMIZATION WORKFLOW:**
        After running this tool, use:
        1. sql_clustering_analyzeClusterStats to identify problematic clusters
        2. sql_clustering_retrieveClusterQueries to get actual SQL from target clusters
        3. LLM analysis to identify patterns and propose specific optimizations

        **USE CASES:**
        - Identify query families consuming the most system resources
        - Find queries with similar patterns but different performance
        - Discover optimization opportunities through clustering analysis
        - Prioritize DBA effort on highest-impact query improvements
        - Understand workload composition and resource distribution

        **EXECUTION TIME:** Typically 5-15 minutes depending on query volume and system resources.

        **PREREQUISITES:**
        - DBC.DBQLSqlTbl and DBC.DBQLOgTbl must be accessible
        - Embedding models and tokenizers must be installed in feature_ext_db
        - Sufficient space in feature_ext_db for intermediate and final tables
        """)
        async def sql_clustering_executeFullPipeline(
            optimal_k: int = Field(default_optimal_k, description=f"Number of clusters to create (default: {default_optimal_k} from config). Typical range: 8-20 depending on workload diversity."),
            max_queries: int = Field(default_max_queries, description=f"Maximum number of top CPU queries to process (default: {default_max_queries}). Larger values provide more comprehensive analysis but require more resources."),
        ) -> ResponseType:
            return execute_db_tool(td.handle_sql_clustering_executeFullPipeline, optimal_k=optimal_k, max_queries=max_queries)

    if config['sql_clustering']['tool']['sql_clustering_analyzeClusterStats']:
        @mcp.tool(description="""
        **ANALYZE SQL QUERY CLUSTER PERFORMANCE STATISTICS**

        This tool analyzes pre-computed cluster statistics to identify optimization opportunities without re-running the clustering pipeline. Perfect for iterative analysis and decision-making on which query clusters to focus optimization efforts.

        **ANALYSIS CAPABILITIES:**
        - **Performance Ranking**: Sort clusters by any performance metric to identify top resource consumers
        - **Resource Impact Assessment**: Compare clusters by CPU usage, I/O volume, and execution complexity
        - **Skew Problem Detection**: Identify clusters with CPU or I/O distribution issues
        - **Workload Characterization**: Understand query patterns by user, application, and workload type
        - **Optimization Prioritization**: Focus on clusters with highest impact potential

        **AVAILABLE SORTING METRICS:**
        - **avg_cpu**: Average CPU seconds per cluster (primary optimization target)
        - **avg_io**: Average logical I/O operations (scan intensity indicator)
        - **avg_cpuskw**: Average CPU skew (distribution problem indicator)
        - **avg_ioskw**: Average I/O skew (hot spot indicator)
        - **avg_pji**: Average Physical-to-Logical I/O ratio (compute intensity)
        - **avg_uii**: Average Unit I/O Intensity (I/O efficiency)
        - **avg_numsteps**: Average query plan complexity
        - **queries**: Number of queries in cluster (frequency indicator)
        - **cluster_silhouette_score**: Clustering quality measure

        **PERFORMANCE CATEGORIZATION:**
        Automatically categorizes clusters using configurable thresholds (from sql_clustering_config.yaml):
        - **HIGH_CPU_USAGE**: Average CPU > config.performance_thresholds.cpu.high
        - **HIGH_IO_USAGE**: Average I/O > config.performance_thresholds.io.high
        - **HIGH_CPU_SKEW**: CPU skew > config.performance_thresholds.skew.high
        - **HIGH_IO_SKEW**: I/O skew > config.performance_thresholds.skew.high
        - **NORMAL**: Clusters within configured normal performance ranges

        **TYPICAL ANALYSIS WORKFLOW:**
        1. Sort by 'avg_cpu' or 'avg_io' to find highest resource consumers
        2. Sort by 'avg_cpuskw' or 'avg_ioskw' to find distribution problems
        4. Use limit_results to focus on top problematic clusters

        **OPTIMIZATION DECISION FRAMEWORK:**
        - **High CPU + High Query Count**: Maximum impact optimization candidates
        - **High Skew + Moderate CPU**: Distribution/statistics problems
        - **High I/O + Low PJI**: Potential indexing opportunities
        - **High NumSteps**: Complex query rewriting candidates

        **OUTPUT FORMAT:**
        Returns detailed cluster statistics with performance rankings, categories, and metadata for LLM analysis and optimization recommendations.
        """)
        async def sql_clustering_analyzeClusterStats(
            sort_by_metric: str = Field("avg_cpu", description="Metric to sort clusters by. Options: avg_cpu, avg_io, avg_cpuskw, avg_ioskw, avg_pji, avg_uii, avg_numsteps, queries, cluster_silhouette_score"),
            limit_results: int = Field(None, description="Limit number of clusters returned (optional). Use to focus on top N problematic clusters."),
        ) -> ResponseType:
            return execute_db_tool(td.handle_sql_clustering_analyzeClusterStats, sort_by_metric=sort_by_metric, limit_results=limit_results)

    if config['sql_clustering']['tool']['sql_clustering_retrieveClusterQueries']:
        @mcp.tool(description="""
        **RETRIEVE ACTUAL SQL QUERIES FROM SPECIFIC CLUSTERS FOR PATTERN ANALYSIS**

        This tool extracts the actual SQL query text and performance metrics from selected clusters, enabling detailed pattern analysis and specific optimization recommendations. Essential for moving from cluster-level analysis to actual query optimization.

        **DETAILED ANALYSIS CAPABILITIES:**
        - **SQL Pattern Recognition**: Analyze actual query structures, joins, predicates, and functions
        - **Performance Correlation**: Connect query patterns to specific performance characteristics
        - **Optimization Identification**: Identify common anti-patterns, missing indexes, inefficient joins
        - **Code Quality Assessment**: Evaluate query construction, complexity, and best practices
        - **Workload Understanding**: See actual business logic and data access patterns

        **QUERY SELECTION STRATEGIES:**
        - **By CPU Impact**: Sort by 'ampcputime' to focus on highest CPU consumers
        - **By I/O Volume**: Sort by 'logicalio' to find scan-intensive queries
        - **By Skew Problems**: Sort by 'cpuskw' or 'ioskw' for distribution issues
        - **By Complexity**: Sort by 'numsteps' for complex execution plans
        - **By Response Time**: Sort by 'response_secs' for user experience impact

        **AVAILABLE METRICS FOR SORTING:**
        - **ampcputime**: Total CPU seconds (primary optimization target)
        - **logicalio**: Total logical I/O operations (scan indicator)
        - **cpuskw**: CPU skew ratio (distribution problems)
        - **ioskw**: I/O skew ratio (hot spot indicators)
        - **pji**: Physical-to-Logical I/O ratio (compute intensity)
        - **uii**: Unit I/O Intensity (I/O efficiency)
        - **numsteps**: Query execution plan steps (complexity)
        - **response_secs**: Wall-clock execution time (user impact)
        - **delaytime**: Time spent in queue (concurrency issues)

        **AUTOMATIC PERFORMANCE CATEGORIZATION:**
        Each query is categorized using configurable thresholds (from sql_clustering_config.yaml):
        - **CPU Categories**: VERY_HIGH_CPU (>config.very_high), HIGH_CPU (>config.high), MEDIUM_CPU (>10s), LOW_CPU
        - **CPU Skew**: SEVERE_CPU_SKEW (>config.severe), HIGH_CPU_SKEW (>config.high), MODERATE_CPU_SKEW (>config.moderate), NORMAL
        - **I/O Skew**: SEVERE_IO_SKEW (>config.severe), HIGH_IO_SKEW (>config.high), MODERATE_IO_SKEW (>config.moderate), NORMAL
        
        Use thresholds set in config file for, CPU - high, very_high, Skew moderate, high, severe

        **TYPICAL OPTIMIZATION WORKFLOW:**
        1. Start with clusters identified from sql_clustering_analyzeClusterStats
        2. Retrieve top queries by impact metric (usually 'ampcputime')
        3. Analyze SQL patterns for common issues:
           - Missing WHERE clauses or inefficient predicates
           - Cartesian products or missing JOIN conditions
           - Inefficient GROUP BY or ORDER BY operations
           - Suboptimal table access patterns
           - Missing or outdated statistics
        4. Develop specific optimization recommendations

        **QUERY LIMIT STRATEGY:**
        - Use the query limit set in config file for  pattern recognition and analysis, unless user specifies a different limit

        **OUTPUT INCLUDES:**
        - Complete SQL query text for each query
        - All performance metrics, user, application, and workload context, cluster membership and rankings
        - Performance categories for quick filtering        
        """)
        async def sql_clustering_retrieveClusterQueries(
            cluster_ids: List[int] = Field(..., description="List of cluster IDs to retrieve queries from (e.g., [2, 5, 8]). Get these from sql_clustering_analyzeClusterStats results."),
            metric: str = Field("ampcputime", description="Performance metric to sort queries by. Options: ampcputime, logicalio, cpuskw, ioskw, pji, uii, numsteps, response_secs, delaytime"),
            limit_per_cluster: int = Field(250, description="Maximum number of top queries to retrieve per cluster"),
        ) -> ResponseType:
            return execute_db_tool(td.handle_sql_clustering_retrieveClusterQueries, cluster_ids=cluster_ids, metric=metric, limit_per_cluster=limit_per_cluster)

    if config['sql_clustering']['prompt']['sql_clustering_optimizationGuidelines']:
        @mcp.prompt()
        async def sql_clustering_optimizationGuidelines() -> UserMessage:
            """Guidelines for analyzing SQL clustering results and providing optimization recommendations."""
            return UserMessage(role="user", content=TextContent(type="text", text=td.handle_sql_clustering_optimizationGuidelines))
        
#------------------ Security Tools  ------------------#

if config['sec']['allmodule']:
    if config['sec']['tool']['sec_userDbPermissions']:
        @mcp.tool(description="Get permissions for a user.")
        async def sec_userDbPermissions(
            user_name: str = Field(description="User name", default=""),
            ) -> ResponseType:
            """Get permissions for a user."""
            return execute_db_tool( td.handle_sec_userDbPermissions, user_name=user_name)

    if config['sec']['tool']['sec_rolePermissions']:
        @mcp.tool(description="Get permissions for a role.")
        async def sec_rolePermissions(
            role_name: str = Field(description="Role name", default=""),
            ) -> ResponseType:
            """Get permissions for a role."""
            return execute_db_tool( td.handle_sec_rolePermissions, role_name=role_name)

    if config['sec']['tool']['sec_userRoles']:
        @mcp.tool(description="Get roles assigned to a user.")
        async def sec_userRoles(
            user_name: str = Field(description="User name", default=""),
            ) -> ResponseType:
            """Get roles assigned to a user."""
            return execute_db_tool( td.handle_sec_userRoles, user_name=user_name)


#------------------ Enterprise Vectore Store Tools  ------------------#

# if config['evs']['allmodule']:
#     if config['evs']['tool']['evs_vectorStoreSimilaritySearch']:
#         @mcp.tool(description="Enterprise Vector Store similarity search")
#         async def evs_vectorStoreSimilaritySearch(
#             question: str = Field(description="Natural language question"),
#             top_k: int = Field(1, description="top matches to return"),
#         ) -> ResponseType:

#             return execute_vs_tool(
#                 td.evs_tools.handle_evs_vectorStoreSimilaritySearch,
#                 question=question,
#                 top_k=top_k,
#             )



#--------------- Feature Store Tools ---------------#
# Feature tools leveraging the tdfs4ds package.

if config['fs']['allmodule']:
    class FeatureStoreConfig(BaseModel):
        """
        Configuration class for the feature store. This model defines the metadata and catalog sources 
        used to organize and access features, processes, and datasets across data domains.
        """

        data_domain: Optional[str] = Field(
            default=None,
            description="The data domain associated with the feature store, grouping features within the same namespace."
        )

        entity: Optional[str] = Field(
            default=None,
            description="The list of entities, comma separated and in alphabetical order, upper case."
        )

        db_name: Optional[str] = Field(
            default=None,
            description="Name of the database where the feature store is hosted."
        )

        feature_catalog: Optional[str] = Field(
            default=None,
            description=(
                "Name of the feature catalog table. "
                "This table contains detailed metadata about features and entities."
            )
        )

        process_catalog: Optional[str] = Field(
            default=None,
            description=(
                "Name of the process catalog table. "
                "Used to retrieve information about feature generation processes, features, and associated entities."
            )
        )

        dataset_catalog: Optional[str] = Field(
            default=None,
            description=(
                "Name of the dataset catalog table. "
                "Used to list and manage available datasets within the feature store."
            )
        )

    fs_config = FeatureStoreConfig() 

    if config['fs']['tool']['fs_reconnect_to_database']:
        @mcp.tool(description="Reconnect to the Teradata database if the connection is lost.")
        async def reconnect_to_database() -> ResponseType:
            """Reconnect to Teradata database if connection is lost."""
            global _tdconn
            try:
                _tdconn = td.TDConn()
                td.teradataml_connection()
                return format_text_response("Reconnected to Teradata database successfully.")
            except Exception as e:
                logger.error(f"Error reconnecting to database: {e}")
                return format_error_response(str(e))

    if config['fs']['tool']['fs_setFeatureStoreConfig']:
        @mcp.tool(description="Set or update the feature store configuration (database and data domain).")
        async def fs_setFeatureStoreConfig(
            data_domain: Optional[str] = None,
            db_name: Optional[str] = None,
            entity: Optional[str] = None,
        ) -> ResponseType:
            if db_name:
                if tdfs4ds.connect(database=db_name):
                    logger.info(f"connected to the feature store of the {db_name} database")
                    # Reset data_domain if DB name changes
                    if not (fs_config.db_name and fs_config.db_name.upper() == db_name.upper()):
                        fs_config.data_domain = None
                    
                    fs_config.db_name = db_name
                    logger.info(f"connected to the feature store of the {db_name} database")
                    fs_config.feature_catalog = f"{db_name}.{tdfs4ds.FEATURE_CATALOG_NAME_VIEW}"
                    logger.info(f"feature catalog {fs_config.feature_catalog}")
                    fs_config.process_catalog = f"{db_name}.{tdfs4ds.PROCESS_CATALOG_NAME_VIEW}"
                    logger.info(f"process catalog {fs_config.process_catalog}")
                    fs_config.dataset_catalog = f"{db_name}.FS_V_FS_DATASET_CATALOG"  # <- fixed line
                    logger.info(f"dataset catalog {fs_config.dataset_catalog}")

            if fs_config.db_name is not None and data_domain is not None:
                sql_query_ = f"SEL count(*) AS N FROM {fs_config.feature_catalog} WHERE UPPER(data_domain) = '{data_domain.upper()}'"
                logger.info(f"{sql_query_}")
                result = tdml.execute_sql(sql_query_)
                logger.info(f"{result}")
                if result.fetchall()[0][0] > 0:
                    fs_config.data_domain = data_domain
                else:
                    fs_config.data_domain = None

            if fs_config.db_name is not None and fs_config.data_domain is not None and entity is not None:
                sql_query_ = f"SEL count(*) AS N FROM {fs_config.feature_catalog} WHERE UPPER(data_domain) = '{data_domain.upper()}' AND ENTITY_NAME = '{entity.upper()}'"
                logger.info(f"{sql_query_}")
                result = tdml.execute_sql(sql_query_)
                logger.info(f"{result}")
                if result.fetchall()[0][0] > 0:
                    fs_config.entity = entity
            return format_text_response(f"Feature store config updated: {fs_config.dict(exclude_none=True)}")

    if config['fs']['tool']['fs_getFeatureStoreConfig']:
        @mcp.tool(description="Display the current feature store configuration (database and data domain).")
        async def fs_getFeatureStoreConfig() -> ResponseType:
            return format_text_response(f"Current feature store config: {fs_config.dict(exclude_none=True)}")

    if config['fs']['tool']['fs_isFeatureStorePresent']:
        @mcp.tool(description=("Check if a feature store is present in the specified database." ))
        async def fs_isFeatureStorePresent(
            db_name: str = Field(..., description="Name of the database to check for a feature store.")
        ) -> ResponseType:
            return execute_db_tool(td.handle_fs_isFeatureStorePresent, db_name=db_name)

    if config['fs']['tool']['fs_featureStoreContent']:
        @mcp.tool(description=("Returns a summary of the feature store content. Use this to understand what data is available in the feature store"))
        async def fs_featureStoreContent(
        ) -> ResponseType:
            return execute_db_tool(td.handle_fs_featureStoreContent, fs_config=fs_config)

    if config['fs']['tool']['fs_getDataDomains']:
        @mcp.tool(description=( "List the available data domains. Requires a configured `db_name`  in the feature store config. Use this to explore which entities can be used when building a dataset."))
        async def fs_getDataDomains(
            entity: str = Field(..., description="The .")
        ) -> ResponseType:
            return execute_db_tool(td.handle_fs_getDataDomains, fs_config=fs_config)

    if config['fs']['tool']['fs_getFeatures']:
        @mcp.tool(description=("List the list of features. Requires a configured `db_name` and  `data_domain` in the feature store config. Use this to explore the features available ."))
        async def fs_getFeatures(
        ) -> ResponseType:
            return execute_db_tool(td.handle_fs_getFeatures, fs_config=fs_config)

    if config['fs']['tool']['fs_getAvailableDatasets']:
        @mcp.tool(description=("List the list of available datasets.Requires a configured `db_name` in the feature store config.Use this to explore the datasets that are available ."))
        async def fs_getAvailableDatasets(
        ) -> ResponseType:
            return execute_db_tool(td.handle_fs_getAvailableDatasets, fs_config=fs_config)

    if config['fs']['tool']['fs_getFeatureDataModel']:
        @mcp.tool(description=("Return the schema of the feature store.Requires a feature store in the configured database (`db_name`)."))
        async def fs_getFeatureDataModel(
        ) -> ResponseType:
            return execute_db_tool(td.handle_fs_getFeatureDataModel, fs_config=fs_config)


    if config['fs']['tool']['fs_getAvailableEntities']:
        @mcp.tool(description=("List the available entities for a given data domain. Requires a configured `db_name` and `data_domain` and  `entity` in the feature store config. Use this to explore which entities can be used when building a dataset."))
        async def fs_getAvailableEntities(
        ) -> ResponseType:
            return execute_db_tool(td.handle_fs_getAvailableEntities, fs_config=fs_config)

    if config['fs']['tool']['fs_createDataset']:
        @mcp.tool( description=("Create a dataset using selected features and an entity from the feature store. The dataset is created in the specified target database under the given name. Requires a configured feature store and data domain. Registers the dataset in the catalog automatically. Use this when you want to build and register a new dataset for analysis or modeling." ) )
        async def fs_createDataset(
            entity_name: str = Field(..., description="Entity for which the dataset will be created. Available entities are reported in the feature catalog."),
            feature_selection: list[str] = Field(..., description="List of features to include in the dataset. Available features are reported in the feature catalog."),
            dataset_name: str = Field(..., description="Name of the dataset to create."),
            target_database: str = Field(..., description="Target database where the dataset will be created.")
        ) -> ResponseType:
            return execute_db_tool(td.handle_fs_createDataset, fs_config=fs_config, entity_name=entity_name, feature_selection=feature_selection, dataset_name=dataset_name, target_database=target_database)


#------------------ Custom Objects  ------------------#
# Custom tools, resources and prompts are defined as SQL queries in a YAML file and loaded at startup.

custom_object_files = [file for file in os.listdir() if file.endswith("_objects.yaml")]
custom_objects = {}
custom_glossary = {}

for file in custom_object_files:
    with open(file) as f:
        loaded = yaml.safe_load(f)
        if loaded:
            custom_objects.update(loaded)  # Merge dictionaries


def make_custom_prompt(prompt_name: str, prompt: str, desc: str):
    async def _dynamic_prompt():
        # SQL is closed over without parameters
        return UserMessage(role="user", content=TextContent(type="text", text=prompt))
    _dynamic_prompt.__name__ = prompt_name
    return mcp.prompt(description=desc)(_dynamic_prompt)

def make_custom_query_tool(name, tool):
    param_defs = tool.get("parameters", {})
    # 1. Build Parameter objects
    parameters = []
    annotations = {}
    # param_defs is now a dict keyed by name
    for param_name, p in param_defs.items():
        type_hint = p.get("type_hint", str)    # e.g. int, float, str, etc.
        default = inspect.Parameter.empty if p.get("required", True) else p.get("default", None)
        parameters.append(
            inspect.Parameter(
                param_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=type_hint
            )
        )
        annotations[param_name] = type_hint

    # 2. Create the new signature
    sig = inspect.Signature(parameters)

    # 3. Define your generic handler
    async def _dynamic_tool(**kwargs):
        """Dynamically generated tool for {name}"""
        missing = [n for n in annotations if n not in kwargs]
        if missing:
            raise ValueError(f"Missing parameters: {missing}")
        return execute_db_tool(td.handle_base_readQuery, tool["sql"], **kwargs)

    # 4. Inject signature & annotations
    _dynamic_tool.__signature__   = sig
    _dynamic_tool.__annotations__ = annotations

    # 5. Register with FastMCP
    return mcp.tool(
        name=name,
        description=tool.get("description", "")
    )(_dynamic_tool)

def generate_cube_query_tool(name, cube):
    """
    Generate a function to create aggregation SQL from a cube definition.

    :param cube: The cube definition
    :return: A SQL query string generator function taking dimensions and measures as comma-separated strings.
    """
    def _cube_query_tool(dimensions: str, measures: str) -> str:
        """
        Generate a SQL query string for the cube using the specified dimensions and measures.

        Args:
            dimensions (str): Comma-separated dimension names (keys in cube['dimensions']).
            measures (str): Comma-separated measure names (keys in cube['measures']).

        Returns:
            str: The generated SQL query.
        """
        dim_list_raw = [d.strip() for d in dimensions.split(",") if d.strip()]
        met_list_raw = [m.strip() for m in measures.split(",") if m.strip()]
        # Get dimension expressions from dictionary
        dim_list = ",\n  ".join([
            cube["dimensions"][d]["expression"] if d in cube["dimensions"] else d
            for d in dim_list_raw
        ])
        met_lines = []
        for measure in met_list_raw:
            mdef = cube["measures"].get(measure)
            if mdef is None:
                raise ValueError(f"Measure '{measure}' not found in cube '{name}'.")
            expr = mdef["expression"]
            met_lines.append(f"{expr} AS {measure}")
        met_block = ",\n  ".join(met_lines)
        sql = (
            "SELECT\n"
            f"  {dim_list},\n"
            f"  {met_block}\n"
            "FROM (\n"
            f"{cube['sql'].strip()}\n"
            ") AS c\n"
            f"GROUP BY {', '.join(dim_list_raw)};"
        )
        return sql
    return _cube_query_tool

def make_custom_cube_tool(name, cube):
    async def _dynamic_tool(dimensions, measures):
        # Accept dimensions and measures as comma-separated strings, parse to lists
        return execute_db_tool(
            td.handle_base_dynamicQuery,
            sql_generator=generate_cube_query_tool(name, cube),
            dimensions=dimensions,
            measures=measures
        )
    _dynamic_tool.__name__ = 'get_cube_' + name
    # Build allowed values and definitions for dimensions and measures
    dim_lines = []
    for name, d in cube.get('dimensions', {}).items():
        dim_lines.append(f"    - {name}: {d.get('description', '')}")
    measure_lines = []
    for name, m in cube.get('measures', {}).items():
        measure_lines.append(f"    - {name}: {m.get('description', '')}")
    _dynamic_tool.__doc__ = f"""
    Tool to query the cube '{name}'.
    {cube.get('description', '')}

    Expected inputs:
        dimensions (str): Comma-separated dimension names to group by. Allowed values:
{chr(10).join(dim_lines)}

        measures (str): Comma-separated measure names to aggregate (must match cube definition). Allowed values:
{chr(10).join(measure_lines)}

    Returns:
        Query result as a formatted response.
    """
    return mcp.tool(description=_dynamic_tool.__doc__)(_dynamic_tool)

# Instantiate custom query tools from YAML
custom_terms = []
for name, obj in custom_objects.items():
    obj_type = obj.get("type")
    if obj_type == "tool":
        fn = make_custom_query_tool(name, obj)
        globals()[name] = fn
        logger.info(f"Created custom tool: {name}")
    elif obj_type == "prompt":
        fn = make_custom_prompt(name, obj["prompt"], obj.get("description", ""))
        globals()[name] = fn
        logger.info(f"Created custom prompt: {name}")
    elif obj_type == "cube":
        fn = make_custom_cube_tool(name, obj)
        globals()[name] = fn
        logger.info(f"Created custom cube: {name}")
    elif obj_type == "glossary":
        # Remove the 'type' key to get just the terms
        custom_glossary = {k: v for k, v in obj.items() if k != "type"}
        logger.info(f"Added custom glossary entries for: {name}.")
    else:
        logger.info(f"Type {obj_type if obj_type else ''} for custom object {name} is {'unknown' if obj_type else 'undefined'}.")

    # Look for additional terms to add to the custom glossary (currently only measures and dimensions in cubes)
    for section in ("measures", "dimensions"):
        if section in obj:
            custom_terms.extend((term, details, name) for term, details in obj[section].items())

# Enrich glossary with terms from tools and cubes
for term, details, tool in custom_terms:
    term_key = term.strip()

    if term_key not in custom_glossary:
        # New glossary entry
        custom_glossary[term_key] = {
            "definition": details.get("description"),
            "synonyms": [],
            "tools": [tool]
        }
    else:
        # Existing glossary term → preserve definition, just add tool if missing
        if "tools" not in custom_glossary[term_key]:
            custom_glossary[term_key]["tools"] = []
        if tool not in custom_glossary[term_key]["tools"]:
            custom_glossary[term_key]["tools"].append(tool)

if custom_glossary:
    # Resource returning the entire glossary
    @mcp.resource("glossary://all")
    def get_glossary() -> ResponseType:
        """List all glossary terms."""
        return custom_glossary

    # Resource returning the entire glossary
    @mcp.resource("glossary://definitions")
    def get_glossary_definitions() -> ResponseType:
        """Returns all glossary terms with definitions."""
        return {term: details["definition"] for term, details in custom_glossary.items()}

    # Resource returning all information about a specific glossary term
    @mcp.resource("glossary://term/{term_name}")
    def get_glossary_term(term_name: str)  -> dict:
        """Return the definition, synonyms and associated tools of a specific glossary term."""
        term = custom_glossary.get(term_name)
        if term:
            return term
        else:
            return {"error": f"Glossary term not found: {term_name}"}

#------------------ Custom Tools  ------------------#
# Custom tools are defined as SQL queries in a YAML file and loaded at startup.

if config['cust']['allmodule']:
    query_defs = []
    custom_tool_files = [file for file in os.listdir() if file.endswith("_tools.yaml")]

    for file in custom_tool_files:
        with open(file) as f:
            query_defs.extend(yaml.safe_load(f))  # Concatenate all query definitions


    def make_custom_prompt(prompt: str, prompt_name: str, desc: str):
        async def _dynamic_prompt():
            # SQL is closed over without parameters
            return UserMessage(role="user", content=TextContent(type="text", text=prompt))
        _dynamic_prompt.__name__ = prompt_name
        return mcp.prompt(description=desc)(_dynamic_prompt)

    def make_custom_query_tool(sql_text: str, tool_name: str, desc: str):
        async def _dynamic_tool():
            # SQL is closed over without parameters
            return execute_db_tool( td.handle_base_readQuery, sql=sql_text)
        _dynamic_tool.__name__ = tool_name
        return mcp.tool(description=desc)(_dynamic_tool)

    # Instantiate custom query tools from YAML
    for q in query_defs:
        if q["type"] == "tool":
            fn = make_custom_query_tool(q["sql"], q["name"], q.get("description", ""))
            globals()[q["name"]] = fn
            logger.info(f"Created custom tool: {q["name"]}")
        elif q["type"] == "prompt":
            fn = make_custom_prompt(q["prompt"], q["name"], q.get("description", "") )
            globals()[q["name"]] = fn
            logger.info(f"Created custom prompt: {q["name"]}")
        else:
            logger.info("Custom yaml type is unnkown.")

#------------------ Main ------------------#
# Main function to start the MCP server
#     Description: Initializes the MCP server and sets up signal handling for graceful shutdown.
#         It creates a connection to the Teradata database and starts the server to listen for incoming requests.
#         The function uses asyncio to manage asynchronous operations and handle signals for shutdown.
#         If an error occurs during initialization, it logs a warning message.
async def main():
    global _tdconn
    
    mcp_transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    logger.info(f"MCP_TRANSPORT: {mcp_transport}")

    # Set up proper shutdown handling
    try:
        loop = asyncio.get_running_loop()
        signals = (signal.SIGTERM, signal.SIGINT)
        for s in signals:
            logger.info(f"Registering signal handler for {s.name}")
            loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s)))
    except NotImplementedError:
        # Windows doesn't support signals properly
        logger.warning("Signal handling not supported on Windows")
        pass
    
    # Start the MCP server
    if mcp_transport == "sse":
        mcp.settings.host = os.getenv("MCP_HOST")
        mcp.settings.port = int(os.getenv("MCP_PORT"))
        logger.info(f"Starting MCP server on {mcp.settings.host}:{mcp.settings.port}")
        await mcp.run_sse_async()
    elif mcp_transport == "streamable-http":
        mcp.settings.host = os.getenv("MCP_HOST")
        mcp.settings.port = int(os.getenv("MCP_PORT"))
        mcp.settings.streamable_http_path = os.getenv("MCP_PATH", "/mcp/")
        logger.info(f"Starting MCP server on {mcp.settings.host}:{mcp.settings.port} with path {mcp.settings.streamable_http_path}")
        await mcp.run_streamable_http_async()
    else:
        logger.info("Starting MCP server on stdin/stdout")
        await mcp.run_stdio_async()    

#------------------ Shutdown ------------------#
# Shutdown function to handle cleanup and exit
#     Arguments: sig (signal.Signals) - signal received for shutdown
#     Description: Cleans up resources and exits the server gracefully.
#         It sets a flag to indicate that shutdown is in progress and logs the received signal.
#         If the shutdown is already in progress, it forces an immediate exit.
#         The function uses os._exit to terminate the process with a specific exit code.
async def shutdown(sig=None):
    """Clean shutdown of the server."""
    global shutdown_in_progress, _tdconn
    
    logger.info("Shutting down server")
    if shutdown_in_progress:
        logger.info("Forcing immediate exit")
        os._exit(1)  # Use immediate process termination instead of sys.exit
    
    _tdconn.close()
    shutdown_in_progress = True
    if sig:
        logger.info(f"Received exit signal {sig.name}")
    os._exit(128 + sig if sig is not None else 0)


#------------------ Entry Point ------------------#
# Entry point for the script
#     Description: This script is designed to be run as a standalone program.
#         It loads environment variables, initializes logging, and starts the MCP server.
#         The main function is called to start the server and handle incoming requests.
#         If an error occurs during execution, it logs the error and exits with a non-zero status code.
if __name__ == "__main__":
    asyncio.run(main())