"""
PostgreSQL implementation for LightRAG storage components.

Environment Variables:
    LIGHTRAG_SKIP_SCHEMA: Set to "1", "true", or "yes" to skip schema validation for faster startup in production
    LIGHTRAG_PRODUCTION: Set to "1", "true", or "yes" to enable production optimizations
    POSTGRES_CONN_TIMEOUT: Connection timeout in seconds (default: 10.0)
    POSTGRES_STMT_CACHE_SIZE: Number of prepared statements to cache (default: 200)

Config File Options (config.ini):
    [postgres]
    skip_schema_validation = true/false  # Skip schema validation
    connection_timeout = 10.0            # Connection timeout in seconds
    statement_cache_size = 200           # Number of prepared statements to cache
"""
import asyncio
from functools import lru_cache
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Union, final
import numpy as np
import configparser

from lightrag.prompt import GRAPH_FIELD_SEP
from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from .performance import performance

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..namespace import NameSpace, is_namespace
from ..utils import logger

import pipmaster as pm

if not pm.is_installed("asyncpg"):
    pm.install("asyncpg")

import asyncpg  # type: ignore
from asyncpg import Pool  # type: ignore

# Get maximum number of graph nodes from environment variable, default is 1000
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))


class PostgreSQLDB:
    # Class-level configuration for faster startup
    SKIP_SCHEMA_VALIDATION = os.environ.get("LIGHTRAG_SKIP_SCHEMA", "").lower() in ("1", "true", "yes")
    PRODUCTION_MODE = os.environ.get("LIGHTRAG_PRODUCTION", "").lower() in ("1", "true", "yes") 
    
    def __init__(self, config: dict[str, Any], **kwargs: Any):
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.user = config.get("user", "postgres")
        self.password = config.get("password", None)
        self.database = config.get("database", "postgres")
        self.workspace = config.get("workspace", "default")
        self.max = 10  # Reduced from 12 for better resource management
        self.min = 2   # Keep some connections alive
        self.pool: Pool | None = None
        self.statement_cache_size = 200  # Cache prepared statements
        self.init_complete = False
        self.tables_checked = False
        self.connection_timeout = 10.0  # seconds
        self.command_timeout = 60.0     # seconds
        
        # Override from config
        self.skip_schema_validation = config.get("skip_schema_validation", self.SKIP_SCHEMA_VALIDATION)
        
        # Connection options for better performance
        self.connection_options = {
            "server_settings": {
                "application_name": "LightRAG",
                "search_path": "public",
                "statement_timeout": "60000",  # milliseconds
                "idle_in_transaction_session_timeout": "300000"  # 5 min
            }
        }

        if self.user is None or self.password is None or self.database is None:
            raise ValueError("Missing database user, password, or database")
            
        if self.skip_schema_validation:
            logger.info("Schema validation is disabled - tables will not be checked")
            self.tables_checked = True

    async def initdb(self):
        try:
            if self.pool is not None and not self.pool._closed:
                logger.debug("Connection pool already exists, reusing")
                return
                
            logger.info(f"Creating connection pool to {self.host}:{self.port}/{self.database}")
            self.pool = await asyncpg.create_pool(  # type: ignore
                user=self.user,
                password=self.password,
                database=self.database,
                host=self.host,
                port=self.port,
                min_size=self.min,
                max_size=self.max,
                command_timeout=self.command_timeout,
                statement_cache_size=self.statement_cache_size,
                timeout=self.connection_timeout,
                **self.connection_options
            )

            logger.info(
                f"PostgreSQL, Connected to database at {self.host}:{self.port}/{self.database}"
            )
            self.init_complete = True
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to connect database at {self.host}:{self.port}/{self.database}, Got:{e}"
            )
            raise

    @staticmethod
    async def configure_age(connection: asyncpg.Connection, graph_name: str) -> None:
        """Set the Apache AGE environment and creates a graph if it does not exist.

        This method:
        - Sets the PostgreSQL `search_path` to include `ag_catalog`, ensuring that Apache AGE functions can be used without specifying the schema.
        - Attempts to create a new graph with the provided `graph_name` if it does not already exist.
        - Silently ignores errors related to the graph already existing.

        """
        try:
            await connection.execute(  # type: ignore
                'SET search_path = ag_catalog, "$user", public'
            )
            await connection.execute(  # type: ignore
                f"select create_graph('{graph_name}')"
            )
        except (
            asyncpg.exceptions.InvalidSchemaNameError,
            asyncpg.exceptions.UniqueViolationError,
        ):
            pass

    async def execute_raw(self, sql: str):
        """Execute SQL directly without prepared statement"""
        if self.pool is None or self.pool._closed:
            await self.initdb()
            
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(sql)
                return True
        except Exception as e:
            logger.error(f"Error executing raw SQL: {e}")
            return False
    
    async def get_existing_tables(self) -> set[str]:
        """Get all existing LightRAG tables in one query"""
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('LIGHTRAG_DOC_FULL', 'LIGHTRAG_DOC_CHUNKS', 'LIGHTRAG_VDB_ENTITY', 
                             'LIGHTRAG_VDB_RELATION', 'LIGHTRAG_LLM_CACHE', 'LIGHTRAG_DOC_STATUS')
        """
        try:
            result = await self.query(query, multirows=True)
            if result:
                return set(r.get('table_name', '').upper() for r in result)
            return set()
        except Exception as e:
            logger.warning(f"Error fetching existing tables: {e}")
            return set()
            
    async def get_existing_indexes(self) -> set[str]:
        """Get all existing LightRAG indexes in one query"""
        query = """
            SELECT indexname
            FROM pg_indexes 
            WHERE indexname LIKE 'idx_%'
            AND (schemaname = 'public' OR schemaname IS NULL)
        """
        try:
            result = await self.query(query, multirows=True)
            if result:
                return set(r.get('indexname', '').lower() for r in result)
            return set()
        except Exception as e:
            logger.warning(f"Error fetching existing indexes: {e}")
            return set()
            
    async def check_tables(self):
        # Skip if tables have already been checked in this instance
        if self.tables_checked:
            logger.debug("Tables already checked, skipping")
            return
            
        logger.info("Checking database tables...")
        start_time = time.time()
            
        # Ensure pgvector extension is available
        try:
            await self.execute_raw("CREATE EXTENSION IF NOT EXISTS vector")
            logger.debug("PostgreSQL, Ensured vector extension is available")
        except Exception as e:
            logger.warning(f"PostgreSQL, Unable to create vector extension: {e}")
        
        # Get all existing tables and indexes in batch queries
        existing_tables = await self.get_existing_tables()
        existing_indexes = await self.get_existing_indexes()
        
        logger.debug(f"Found existing tables: {existing_tables}")
        logger.debug(f"Found existing indexes: {existing_indexes}")
        
        # Process missing tables
        for table_name in TABLES.keys():
            try:
                if table_name in existing_tables:
                    # Table exists, only ensure indexes that don't exist
                    needed_indexes = self._get_missing_indexes(table_name, existing_indexes)
                    if needed_indexes:
                        await self._create_indexes(table_name, needed_indexes)
                else:
                    # Table needs to be created
                    logger.info(f"Creating table {table_name}")
                    
                    # Split DDL into individual statements
                    ddl_statements = self._split_ddl(TABLES[table_name]["ddl"])
                    
                    # Execute each statement separately
                    for stmt in ddl_statements:
                        if stmt.strip():
                            await self.execute_raw(stmt)
                            
                    logger.info(f"Successfully created table {table_name}")
            except Exception as e:
                logger.error(f"Error with table {table_name}: {e}")
                
        # Mark tables as checked
        self.tables_checked = True
        logger.info(f"Database tables check completed in {time.time() - start_time:.2f}s")
        
    def _split_ddl(self, ddl: str) -> list[str]:
        """Split DDL with multiple statements into individual commands"""
        # First, find the main CREATE TABLE statement
        table_stmt_end = ddl.find(';', ddl.find('CREATE TABLE'))
        if table_stmt_end == -1:
            # No semicolon found, assume entire string is one statement
            return [ddl]
            
        # Extract the create table part
        create_table_stmt = ddl[:table_stmt_end+1]
        remaining = ddl[table_stmt_end+1:]
        
        # Split remaining statements by semicolons with comments handling
        statements = []
        statements.append(create_table_stmt)
        
        # Simple statement splitter that respects comments
        current_stmt = ""
        in_comment = False
        for line in remaining.split("\n"):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Handle comment lines
            if line.startswith("--"):
                continue
                
            # Add this line to current statement
            current_stmt += line + " "
            
            # If line ends with semicolon, it's a complete statement
            if line.endswith(";"):
                statements.append(current_stmt.strip())
                current_stmt = ""
                
        # Add any final statement without semicolon
        if current_stmt.strip():
            statements.append(current_stmt.strip())
            
        return statements
                    
    # Define all indexes in one place as a class variable for easy access
    TABLE_INDEXES = {
        "LIGHTRAG_DOC_CHUNKS": [
            {
                "name": "idx_doc_chunks_full_doc_id",
                "sql": "CREATE INDEX IF NOT EXISTS idx_doc_chunks_full_doc_id ON LIGHTRAG_DOC_CHUNKS(workspace, full_doc_id)",
                "is_vector": False
            },
            {
                "name": "idx_chunks_vector",
                "sql": "CREATE INDEX IF NOT EXISTS idx_chunks_vector ON LIGHTRAG_DOC_CHUNKS USING ivfflat (content_vector vector_cosine_ops) WITH (lists = 100)",
                "is_vector": True
            }
        ],
        "LIGHTRAG_VDB_ENTITY": [
            {
                "name": "idx_entity_entity_name",
                "sql": "CREATE INDEX IF NOT EXISTS idx_entity_entity_name ON LIGHTRAG_VDB_ENTITY(workspace, entity_name)",
                "is_vector": False
            },
            {
                "name": "idx_entity_vector",
                "sql": "CREATE INDEX IF NOT EXISTS idx_entity_vector ON LIGHTRAG_VDB_ENTITY USING ivfflat (content_vector vector_cosine_ops) WITH (lists = 100)",
                "is_vector": True
            }
        ],
        "LIGHTRAG_VDB_RELATION": [
            {
                "name": "idx_relation_source_id",
                "sql": "CREATE INDEX IF NOT EXISTS idx_relation_source_id ON LIGHTRAG_VDB_RELATION(workspace, source_id)",
                "is_vector": False
            },
            {
                "name": "idx_relation_target_id",
                "sql": "CREATE INDEX IF NOT EXISTS idx_relation_target_id ON LIGHTRAG_VDB_RELATION(workspace, target_id)",
                "is_vector": False
            },
            {
                "name": "idx_relation_vector",
                "sql": "CREATE INDEX IF NOT EXISTS idx_relation_vector ON LIGHTRAG_VDB_RELATION USING ivfflat (content_vector vector_cosine_ops) WITH (lists = 100)",
                "is_vector": True
            }
        ],
        "LIGHTRAG_LLM_CACHE": [
            {
                "name": "idx_llm_cache_mode",
                "sql": "CREATE INDEX IF NOT EXISTS idx_llm_cache_mode ON LIGHTRAG_LLM_CACHE(workspace, mode)",
                "is_vector": False
            }
        ],
        "LIGHTRAG_DOC_STATUS": [
            {
                "name": "idx_doc_status_status",
                "sql": "CREATE INDEX IF NOT EXISTS idx_doc_status_status ON LIGHTRAG_DOC_STATUS(workspace, status)",
                "is_vector": False
            }
        ]
    }
    
    def _get_missing_indexes(self, table_name: str, existing_indexes: set[str]) -> list[dict]:
        """Determine which indexes need to be created for a table"""
        if table_name not in self.TABLE_INDEXES:
            return []
            
        # Find indexes for this table that don't exist yet
        missing_indexes = []
        for index_info in self.TABLE_INDEXES[table_name]:
            if index_info["name"].lower() not in existing_indexes:
                missing_indexes.append(index_info)
                
        return missing_indexes
    
    async def _create_indexes(self, table_name: str, indexes: list[dict]) -> None:
        """Create multiple indexes for a table"""
        if not indexes:
            return
            
        logger.debug(f"Creating {len(indexes)} missing indexes for {table_name}")
        
        for index_info in indexes:
            try:
                await self.execute_raw(index_info["sql"])
                logger.debug(f"Created index {index_info['name']}")
            except Exception as e:
                # Vector indexes might fail if pgvector isn't set up properly
                if index_info["is_vector"]:
                    logger.warning(f"Could not create vector index {index_info['name']}: {e}")
                else:
                    logger.warning(f"Failed to create index {index_info['name']}: {e}")
    
    # Legacy method for backward compatibility
    async def _ensure_indexes(self, table_name: str):
        """Ensures required indexes exist on existing tables"""
        try:
            # Get existing indexes in one query
            existing_indexes = await self.get_existing_indexes()
            
            # Find which indexes need to be created
            needed_indexes = self._get_missing_indexes(table_name, existing_indexes)
            
            # Create any missing indexes
            if needed_indexes:
                await self._create_indexes(table_name, needed_indexes)
                
        except Exception as e:
            logger.warning(f"Failed to ensure indexes for {table_name}: {e}")
            # Non-fatal - indexes improve performance but aren't essential for functionality

    # Statement cache to avoid repeated preparation
    _stmt_cache = {}
    _stmt_cache_limit = 300
    
    async def _get_prepared_stmt(self, connection, sql):
        """Get or create a prepared statement with caching"""
        # Use hash of SQL as cache key
        cache_key = hash(sql)
        
        if cache_key in self._stmt_cache:
            # Return cached statement if connection matches
            cached_stmt, cached_conn = self._stmt_cache[cache_key]
            if cached_conn is connection:
                return cached_stmt
        
        # Prepare new statement
        stmt = await connection.prepare(sql)
        
        # Manage cache size
        if len(self._stmt_cache) >= self._stmt_cache_limit:
            # Remove a random entry to avoid cache growing too large
            self._stmt_cache.pop(next(iter(self._stmt_cache)))
            
        # Cache the statement with its connection
        self._stmt_cache[cache_key] = (stmt, connection)
        return stmt
    
    async def query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        multirows: bool = False,
        with_age: bool = False,
        graph_name: str | None = None,
    ) -> dict[str, Any] | None | list[dict[str, Any]]:
        if self.pool is None or self.pool._closed:
            await self.initdb()  # Auto-reconnect if needed
        
        async with self.pool.acquire() as connection:  # type: ignore
            if with_age and graph_name:
                await self.configure_age(connection, graph_name)  # type: ignore
            elif with_age and not graph_name:
                raise ValueError("Graph name is required when with_age is True")

            try:
                # Use cached or prepare new statement
                stmt = await self._get_prepared_stmt(connection, sql)
                
                # Execute query
                if params:
                    param_values = []
                    for _, val in params.items():
                        if isinstance(val, list) and all(isinstance(x, str) for x in val):
                            param_values.append(val)
                        else:
                            param_values.append(val)
                    
                    rows = await stmt.fetch(*param_values)
                else:
                    rows = await stmt.fetch()

                # Process results
                if not rows:
                    return [] if multirows else None
                    
                if multirows:
                    columns = [col for col in rows[0].keys()]
                    data = [dict(zip(columns, row)) for row in rows]
                else:
                    columns = rows[0].keys()
                    data = dict(zip(columns, rows[0]))
                    
                return data
            except Exception as e:
                logger.error(f"PostgreSQL database query error: {e}, SQL: {sql[:100]}...")
                raise

    async def execute(
        self,
        sql: str,
        data: dict[str, Any] | None = None,
        upsert: bool = False,
        with_age: bool = False,
        graph_name: str | None = None,
    ):
        if self.pool is None or self.pool._closed:
            await self.initdb()  # Auto-reconnect if needed
        
        try:
            async with self.pool.acquire() as connection:  # type: ignore
                if with_age and graph_name:
                    await self.configure_age(connection, graph_name)  # type: ignore
                elif with_age and not graph_name:
                    raise ValueError("Graph name is required when with_age is True")

                # Use cached or prepare new statement
                stmt = await self._get_prepared_stmt(connection, sql)
                
                # Execute statement
                if data is None:
                    await stmt.fetch()
                else:
                    param_values = []
                    for val in data.values():
                        if isinstance(val, list) and all(isinstance(x, str) for x in val):
                            param_values.append(val)
                        else:
                            param_values.append(val)
                            
                    await stmt.fetch(*param_values)
        except (
            asyncpg.exceptions.UniqueViolationError,
            asyncpg.exceptions.DuplicateTableError,
        ) as e:
            if upsert:
                logger.debug("Key value duplicate, but upsert succeeded.")
            else:
                logger.error(f"Upsert error: {e}")
        except asyncpg.exceptions.QueryCanceledError as e:
            logger.error(f"Query timed out: {sql[:100]}...")
            raise asyncpg.exceptions.QueryCanceledError(f"Query execution timeout: {e}") from e
        except Exception as e:
            logger.error(f"PostgreSQL database execute error: {e}, SQL: {sql[:100]}...")
            raise


class ClientManager:
    _instances: dict[str, Any] = {"db": None, "ref_count": 0}
    _lock = asyncio.Lock()
    _client_init_task = None
    _config = None
    _lazy_init_done = False
    _pool_timeout = 60.0  # Keep connections alive longer (seconds)
    _connection_timeout = 5.0  # seconds
    _keepalive_task = None

    @staticmethod
    def get_config() -> dict[str, Any]:
        # Cache config to avoid repeated file reads
        if ClientManager._config is not None:
            return ClientManager._config
            
        config = configparser.ConfigParser()
        try:
            config.read("config.ini", "utf-8")
        except Exception as e:
            logger.warning(f"Could not read config.ini: {e}, using environment variables only")
            
        # Core database connection settings
        ClientManager._config = {
            "host": os.environ.get(
                "POSTGRES_HOST",
                config.get("postgres", "host", fallback="localhost"),
            ),
            "port": os.environ.get(
                "POSTGRES_PORT", config.get("postgres", "port", fallback=5432)
            ),
            "user": os.environ.get(
                "POSTGRES_USER", config.get("postgres", "user", fallback=None)
            ),
            "password": os.environ.get(
                "POSTGRES_PASSWORD",
                config.get("postgres", "password", fallback=None),
            ),
            "database": os.environ.get(
                "POSTGRES_DATABASE",
                config.get("postgres", "database", fallback=None),
            ),
            "workspace": os.environ.get(
                "POSTGRES_WORKSPACE",
                config.get("postgres", "workspace", fallback="default"),
            ),
            # Schema validation control
            "skip_schema_validation": os.environ.get("LIGHTRAG_SKIP_SCHEMA", "").lower() in ("1", "true", "yes")
                or config.get("postgres", "skip_schema_validation", fallback="").lower() in ("1", "true", "yes"),
            # Performance settings
            "connection_timeout": float(os.environ.get("POSTGRES_CONN_TIMEOUT", 
                                     config.get("postgres", "connection_timeout", fallback="10.0"))),
            "statement_cache_size": int(os.environ.get("POSTGRES_STMT_CACHE_SIZE",
                                      config.get("postgres", "statement_cache_size", fallback="200"))),
        }
        return ClientManager._config

    @classmethod
    async def _keep_connection_alive(cls):
        """Background task to keep connection pool alive"""
        try:
            while True:
                # Use shorter sleep interval so the task can be cancelled promptly
                for _ in range(6):  # 6 x 5s = 30s total
                    await asyncio.sleep(5)
                    # Check if we're shutting down during sleep
                    if cls._instances["db"] is None:
                        logger.debug("Shutting down, stopping keepalive")
                        return
                
                async with cls._lock:
                    if cls._instances["db"] is None:
                        # Database connection gone, stop the keepalive task
                        logger.debug("Database connection gone, stopping keepalive")
                        return
                        
                    if hasattr(cls._instances["db"], "pool") and not cls._instances["db"].pool._closed:
                        # Send a simple query to keep the connection alive
                        try:
                            await cls._instances["db"].execute("SELECT 1")
                            logger.debug("Connection keepalive ping successful")
                        except Exception as e:
                            logger.warning(f"Keepalive ping failed: {e}")
        except asyncio.CancelledError:
            logger.debug("Keepalive task cancelled")
            # Clean self-reference to help with garbage collection
            cls._keepalive_task = None
        except Exception as e:
            logger.error(f"Error in keepalive task: {e}")
            # Clean self-reference
            cls._keepalive_task = None

    @classmethod
    async def _lazy_init(cls):
        """Initialize database client lazily"""
        if cls._lazy_init_done:
            return
            
        config = ClientManager.get_config()
        db = PostgreSQLDB(config)
        
        logger.info("Lazy initializing database client and pool")
        await db.initdb()
        await db.check_tables()
        
        cls._instances["db"] = db
        cls._instances["ref_count"] = 1
        cls._lazy_init_done = True
        
        # Start the keepalive task if it's not running
        if cls._keepalive_task is None or cls._keepalive_task.done():
            cls._keepalive_task = asyncio.create_task(cls._keep_connection_alive())
        
        return db

    @classmethod
    @performance
    async def get_client(cls) -> PostgreSQLDB:
        """Get a client from the shared pool with optimized connection handling"""
        # Configure timeout and retries
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                async with cls._lock:
                    # Check if we already have a valid instance - fast path
                    if cls._instances["db"] is not None and hasattr(cls._instances["db"], "pool") and not cls._instances["db"].pool._closed:
                        cls._instances["ref_count"] += 1
                        return cls._instances["db"]
                    
                    # Initialize if not already done
                    if not cls._lazy_init_done:
                        db = await cls._lazy_init()
                        return db
                    
                    # Re-initialize if needed (pool was closed)
                    logger.info("Reconnecting database client")
                    config = ClientManager.get_config()
                    db = PostgreSQLDB(config)
                    await db.initdb()
                    # Skip table check for reconnection since it's expensive
                    cls._instances["db"] = db
                    cls._instances["ref_count"] = 1
                    return db
                    
            except Exception as e:
                logger.error(f"Error getting client (attempt {retry_count+1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                await asyncio.sleep(0.5 * retry_count)  # Exponential backoff
        
        raise RuntimeError("Failed to get database client after multiple retries")

    @classmethod
    async def release_client(cls, db: PostgreSQLDB):
        """Manage reference count but keep connection pool alive"""
        async with cls._lock:
            if db is not None and db is cls._instances["db"]:
                cls._instances["ref_count"] -= 1
                logger.debug(f"Released client, ref count: {cls._instances['ref_count']}")
                
                # Keep pool alive longer, only close if program is shutting down or ref count has been 0 for a while
                if cls._instances["ref_count"] <= 0:
                    cls._instances["ref_count"] = 0  # Ensure we don't go negative
                    
                    # Program exiting - check if we should cleanup fully
                    if os.environ.get("LIGHTRAG_CLEANUP_ON_EXIT", "").lower() in ("1", "true", "yes"):
                        logger.info("Cleaning up database connections on exit")
                        
                        # Cancel keepalive task if running
                        if cls._keepalive_task and not cls._keepalive_task.done():
                            cls._keepalive_task.cancel()
                            
                        # Close pool
                        if db.pool and not db.pool._closed:
                            await db.pool.close()
                            logger.info("Closed PostgreSQL connection pool")
                            
                        # Clean instance reference
                        cls._instances["db"] = None


@final
@dataclass
class PGKVStorage(BaseKVStorage):
    db: PostgreSQLDB = field(default=None)

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    ################ QUERY METHODS ################

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get doc_full data by id."""
        sql = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.db.workspace, "id": id}
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            array_res = await self.db.query(sql, params, multirows=True)
            res = {}
            for row in array_res:
                res[row["id"]] = row
            return res if res else None
        else:
            response = await self.db.query(sql, params)
            return response if response else None

    async def get_by_mode_and_id(self, mode: str, id: str) -> Union[dict, None]:
        """Specifically for llm_response_cache."""
        sql = SQL_TEMPLATES["get_by_mode_id_" + self.namespace]
        params = {"workspace": self.db.workspace, mode: mode, "id": id}
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            array_res = await self.db.query(sql, params, multirows=True)
            res = {}
            for row in array_res:
                res[row["id"]] = row
            return res
        else:
            return None

    # Query by id
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get doc_chunks data by id"""
        if not ids:
            return []
            
        if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            sql = SQL_TEMPLATES["get_by_ids_full_docs"]
        elif is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            sql = SQL_TEMPLATES["get_by_ids_text_chunks"]
        elif is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            sql = SQL_TEMPLATES["get_by_ids_llm_response_cache"]
        else:
            logger.error(f"Unknown namespace for get_by_ids: {self.namespace}")
            return []
            
        # Properly cast parameters to their PostgreSQL types
        params = {"workspace": str(self.db.workspace), "ids": list(ids)}
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            array_res = await self.db.query(sql, params, multirows=True)
            modes = set()
            dict_res: dict[str, dict] = {}
            for row in array_res:
                modes.add(row["mode"])
            for mode in modes:
                if mode not in dict_res:
                    dict_res[mode] = {}
            for row in array_res:
                dict_res[row["mode"]][row["id"]] = row
            return [{k: v} for k, v in dict_res.items()]
        else:
            return await self.db.query(sql, params, multirows=True)

    async def get_by_status(self, status: str) -> Union[list[dict[str, Any]], None]:
        """Specifically for llm_response_cache."""
        SQL = SQL_TEMPLATES["get_by_status_" + self.namespace]
        params = {"workspace": self.db.workspace, "status": status}
        return await self.db.query(SQL, params, multirows=True)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out duplicated content"""
        if not keys:
            return set()
            
        table_name = namespace_to_table_name(self.namespace)
        sql = f"SELECT id FROM {table_name} WHERE workspace=$1::varchar(255) AND id = ANY($2::varchar(255)[])"
        params = {"workspace": str(self.db.workspace), "ids": list(keys)}
        
        try:
            res = await self.db.query(sql, params, multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            new_keys = set([s for s in keys if s not in exist_keys])
            return new_keys
        except Exception as e:
            logger.error(
                f"PostgreSQL database,\nsql:{sql},\nparams:{params},\nerror:{e}"
            )
            raise

    ################ INSERT METHODS ################
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            pass
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            for k, v in data.items():
                upsert_sql = SQL_TEMPLATES["upsert_doc_full"]
                _data = {
                    "id": k,
                    "content": v["content"],
                    "workspace": self.db.workspace,
                }
                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            for mode, items in data.items():
                for k, v in items.items():
                    upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
                    _data = {
                        "workspace": self.db.workspace,
                        "id": k,
                        "original_prompt": v["original_prompt"],
                        "return_value": v["return"],
                        "mode": mode,
                    }

                    await self.db.execute(upsert_sql, _data)

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"Unknown namespace for deletion: {self.namespace}")
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"

        try:
            await self.db.execute(
                delete_sql, {"workspace": self.db.workspace, "ids": ids}
            )
            logger.debug(
                f"Successfully deleted {len(ids)} records from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"Error while deleting records from {self.namespace}: {e}")

    async def drop_cache_by_modes(self, modes: list[str] | None = None) -> bool:
        """Delete specific records from storage by cache mode

        Args:
            modes (list[str]): List of cache modes to be dropped from storage

        Returns:
            bool: True if successful, False otherwise
        """
        if not modes:
            return False

        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return False

            if table_name != "LIGHTRAG_LLM_CACHE":
                return False

            sql = f"""
            DELETE FROM {table_name}
            WHERE workspace = $1 AND mode = ANY($2)
            """
            params = {"workspace": self.db.workspace, "modes": modes}

            logger.info(f"Deleting cache by modes: {modes}")
            await self.db.execute(sql, params)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache by modes {modes}: {e}")
            return False

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specifiy_table_workspace"].format(
                table_name=table_name
            )
            await self.db.execute(drop_sql, {"workspace": self.db.workspace})
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@final
@dataclass
class PGVectorStorage(BaseVectorStorage):
    db: PostgreSQLDB | None = field(default=None)

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = config.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    def _upsert_chunks(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        try:
            upsert_sql = SQL_TEMPLATES["upsert_chunk"]
            data: dict[str, Any] = {
                "workspace": self.db.workspace,
                "id": item["__id__"],
                "tokens": item["tokens"],
                "chunk_order_index": item["chunk_order_index"],
                "full_doc_id": item["full_doc_id"],
                "content": item["content"],
                "content_vector": json.dumps(item["__vector__"].tolist()),
                "file_path": item["file_path"],
            }
        except Exception as e:
            logger.error(f"Error to prepare upsert,\nsql: {e}\nitem: {item}")
            raise

        return upsert_sql, data

    def _upsert_entities(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        upsert_sql = SQL_TEMPLATES["upsert_entity"]
        source_id = item["source_id"]
        if isinstance(source_id, str) and GRAPH_FIELD_SEP in source_id:
            chunk_ids = source_id.split(GRAPH_FIELD_SEP)
        else:
            chunk_ids = [source_id]

        data: dict[str, Any] = {
            "workspace": self.db.workspace,
            "id": item["__id__"],
            "entity_name": item["entity_name"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
            "chunk_ids": chunk_ids,
            "file_path": item["file_path"],
            # TODO: add document_id
        }
        return upsert_sql, data

    def _upsert_relationships(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        upsert_sql = SQL_TEMPLATES["upsert_relationship"]
        source_id = item["source_id"]
        if isinstance(source_id, str) and GRAPH_FIELD_SEP in source_id:
            chunk_ids = source_id.split(GRAPH_FIELD_SEP)
        else:
            chunk_ids = [source_id]

        data: dict[str, Any] = {
            "workspace": self.db.workspace,
            "id": item["__id__"],
            "source_id": item["src_id"],
            "target_id": item["tgt_id"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
            "chunk_ids": chunk_ids,
            "file_path": item["file_path"],
            # TODO: add document_id
        }
        return upsert_sql, data

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        current_time = time.time()
        list_data = [
            {
                "__id__": k,
                "__created_at__": current_time,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        for item in list_data:
            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                upsert_sql, data = self._upsert_chunks(item)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                upsert_sql, data = self._upsert_entities(item)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                upsert_sql, data = self._upsert_relationships(item)
            else:
                raise ValueError(f"{self.namespace} is not supported")

            await self.db.execute(upsert_sql, data)

    #################### query method ###############
    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        embeddings = await self.embedding_func([query])
        embedding = embeddings[0]
        embedding_string = ",".join(map(str, embedding))

        if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
            sql = SQL_TEMPLATES["vector_query_chunks"]
        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
            sql = SQL_TEMPLATES["vector_query_entities"]
        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
            sql = SQL_TEMPLATES["vector_query_relationships"]
        else:
            raise ValueError(f"Unsupported namespace for vector query: {self.namespace}")
        
        params = {
            "workspace": str(self.db.workspace),
            "top_k": int(top_k),
            "doc_ids": list(ids) if ids is not None else None,
            "embedding": f"[{embedding_string}]"
        }

        try:
            results = await self.db.query(sql, params=params, multirows=True)
            
            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                for result in results:
                    if "entity_name" not in result and "e.entity_name" in result:
                        result["entity_name"] = result["e.entity_name"]
                    # Ensure source_id is present
                    if "source_id" not in result:
                        result["source_id"] = result.get("chunk_ids", ["unknown"])[0] if isinstance(result.get("chunk_ids", []), list) else "unknown"
            
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                for result in results:
                    if "src_id" not in result and "r.source_id" in result:
                        result["src_id"] = result["r.source_id"]
                    if "tgt_id" not in result and "r.target_id" in result:
                        result["tgt_id"] = result["r.target_id"]
                    # Ensure source_id is present
                    if "source_id" not in result:
                        result["source_id"] = result.get("chunk_ids", ["unknown"])[0] if isinstance(result.get("chunk_ids", []), list) else "unknown"
            
            return results
        except Exception as e:
            logger.error(f"Error in vector query: {e}")
            logger.error(f"Query was: {sql}")
            logger.error(f"Params were: {params}")
            return []

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs from the storage.

        Args:
            ids: List of vector IDs to be deleted
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"Unknown namespace for vector deletion: {self.namespace}")
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"

        try:
            await self.db.execute(
                delete_sql, {"workspace": self.db.workspace, "ids": ids}
            )
            logger.debug(
                f"Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"Error while deleting vectors from {self.namespace}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by its name from the vector storage.

        Args:
            entity_name: The name of the entity to delete
        """
        try:
            # Construct SQL to delete the entity
            delete_sql = """DELETE FROM LIGHTRAG_VDB_ENTITY
                            WHERE workspace=$1 AND entity_name=$2"""

            await self.db.execute(
                delete_sql, {"workspace": self.db.workspace, "entity_name": entity_name}
            )
            logger.debug(f"Successfully deleted entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity.

        Args:
            entity_name: The name of the entity whose relations should be deleted
        """
        try:
            # Delete relations where the entity is either the source or target
            delete_sql = """DELETE FROM LIGHTRAG_VDB_RELATION
                            WHERE workspace=$1 AND (source_id=$2 OR target_id=$2)"""

            await self.db.execute(
                delete_sql, {"workspace": self.db.workspace, "entity_name": entity_name}
            )
            logger.debug(f"Successfully deleted relations for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for entity {entity_name}: {e}")

    async def search_by_prefix(self, prefix: str) -> list[dict[str, Any]]:
        """Search for records with IDs starting with a specific prefix.

        Args:
            prefix: The prefix to search for in record IDs

        Returns:
            List of records with matching ID prefixes
        """
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"Unknown namespace for prefix search: {self.namespace}")
            return []

        search_sql = f"SELECT * FROM {table_name} WHERE workspace=$1 AND id LIKE $2"
        params = {"workspace": self.db.workspace, "prefix": f"{prefix}%"}

        try:
            results = await self.db.query(search_sql, params, multirows=True)
            logger.debug(f"Found {len(results)} records with prefix '{prefix}'")

            # Format results to match the expected return format
            formatted_results = []
            for record in results:
                formatted_record = dict(record)
                # Ensure id field is available (for consistency with NanoVectorDB implementation)
                if "id" not in formatted_record:
                    formatted_record["id"] = record["id"]
                formatted_results.append(formatted_record)

            return formatted_results
        except Exception as e:
            logger.error(f"Error during prefix search for '{prefix}': {e}")
            return []

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"Unknown namespace for ID lookup: {self.namespace}")
            return None

        query = f"SELECT * FROM {table_name} WHERE workspace=$1 AND id=$2"
        params = {"workspace": self.db.workspace, "id": id}

        try:
            result = await self.db.query(query, params)
            if result:
                return dict(result)
            return None
        except Exception as e:
            logger.error(f"Error retrieving vector data for ID {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"Unknown namespace for IDs lookup: {self.namespace}")
            return []

        # Use explicit varchar(255) type for proper index optimization
        query = f"SELECT * FROM {table_name} WHERE workspace=$1::varchar(255) AND id = ANY($2::varchar(255)[])"
        params = {"workspace": str(self.db.workspace), "ids": list(ids)}

        try:
            results = await self.db.query(query, params, multirows=True)
            return [dict(record) for record in results]
        except Exception as e:
            logger.error(f"Error retrieving vector data for IDs {ids}: {e}")
            return []

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specifiy_table_workspace"].format(
                table_name=table_name
            )
            await self.db.execute(drop_sql, {"workspace": self.db.workspace})
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@final
@dataclass
class PGDocStatusStorage(DocStatusStorage):
    db: PostgreSQLDB = field(default=None)

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out duplicated content"""
        if not keys:
            return set()
            
        table_name = namespace_to_table_name(self.namespace)
        sql = f"SELECT id FROM {table_name} WHERE workspace=$1::varchar(255) AND id = ANY($2::varchar(255)[])"
        params = {"workspace": str(self.db.workspace), "ids": list(keys)}
        
        try:
            res = await self.db.query(sql, params, multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            new_keys = set([s for s in keys if s not in exist_keys])
            return new_keys
        except Exception as e:
            logger.error(
                f"PostgreSQL database,\nsql:{sql},\nparams:{params},\nerror:{e}"
            )
            raise

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=$1::varchar(255) and id=$2::varchar(255)"
        params = {"workspace": str(self.db.workspace), "id": id}
        result = await self.db.query(sql, params, True)
        if result is None or result == []:
            return None
        else:
            return dict(
                content=result[0]["content"],
                content_length=result[0]["content_length"],
                content_summary=result[0]["content_summary"],
                status=result[0]["status"],
                chunks_count=result[0]["chunks_count"],
                created_at=result[0]["created_at"],
                updated_at=result[0]["updated_at"],
                file_path=result[0]["file_path"],
            )

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get doc_chunks data by multiple IDs."""
        if not ids:
            return []

        # Explicitly use varchar(255) type casting for optimization
        sql = "SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1::varchar(255) AND id = ANY($2::varchar(255)[])"
        params = {"workspace": str(self.db.workspace), "ids": list(ids)}

        try:
            results = await self.db.query(sql, params, True)
            if not results:
                return []
        except Exception as e:
            logger.error(f"Error retrieving doc status for IDs {ids}: {e}")
            return []
        return [
            {
                "content": row["content"],
                "content_length": row["content_length"],
                "content_summary": row["content_summary"],
                "status": row["status"],
                "chunks_count": row["chunks_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "file_path": row["file_path"],
            }
            for row in results
        ]

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        sql = """SELECT status as "status", COUNT(1) as "count"
                   FROM LIGHTRAG_DOC_STATUS
                  WHERE workspace=$1::varchar(255) GROUP BY STATUS
                 """
        result = await self.db.query(sql, {"workspace": str(self.db.workspace)}, True)
        counts = {}
        for doc in result:
            counts[doc["status"]] = doc["count"]
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """all documents with a specific status"""
        sql = "SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1::varchar(255) AND status=$2::varchar(64)"
        params = {"workspace": str(self.db.workspace), "status": status.value}
        result = await self.db.query(sql, params, True)
        docs_by_status = {
            element["id"]: DocProcessingStatus(
                content=element["content"],
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=element["created_at"],
                updated_at=element["updated_at"],
                chunks_count=element["chunks_count"],
                file_path=element["file_path"],
            )
            for element in result
        }
        return docs_by_status

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"Unknown namespace for deletion: {self.namespace}")
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"

        try:
            await self.db.execute(
                delete_sql, {"workspace": self.db.workspace, "ids": ids}
            )
            logger.debug(
                f"Successfully deleted {len(ids)} records from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"Error while deleting records from {self.namespace}: {e}")

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Update or insert document status

        Args:
            data: dictionary of document IDs and their status data
        """
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        sql = """insert into LIGHTRAG_DOC_STATUS(workspace,id,content,content_summary,content_length,chunks_count,status,file_path)
                 values($1,$2,$3,$4,$5,$6,$7,$8)
                  on conflict(id,workspace) do update set
                  content = EXCLUDED.content,
                  content_summary = EXCLUDED.content_summary,
                  content_length = EXCLUDED.content_length,
                  chunks_count = EXCLUDED.chunks_count,
                  status = EXCLUDED.status,
                  file_path = EXCLUDED.file_path,
                  updated_at = CURRENT_TIMESTAMP"""
        for k, v in data.items():
            # chunks_count is optional
            await self.db.execute(
                sql,
                {
                    "workspace": self.db.workspace,
                    "id": k,
                    "content": v["content"],
                    "content_summary": v["content_summary"],
                    "content_length": v["content_length"],
                    "chunks_count": v["chunks_count"] if "chunks_count" in v else -1,
                    "status": v["status"],
                    "file_path": v["file_path"],
                },
            )

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specifiy_table_workspace"].format(
                table_name=table_name
            )
            await self.db.execute(drop_sql, {"workspace": self.db.workspace})
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


class PGGraphQueryException(Exception):
    """Exception for the AGE queries."""

    def __init__(self, exception: Union[str, dict[str, Any]]) -> None:
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


@final
@dataclass
class PGGraphStorage(BaseGraphStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use LRU caches for frequent operations
        self._node_cache = {}
        self._edge_cache = {}
        self._node_edges_cache = {}
        self._max_cache_size = 1000  # Configurable
        self._cache_hits = 0
        self._cache_misses = 0
        self.graph_name = self.namespace or os.environ.get("AGE_GRAPH_NAME", "lightrag")
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }
        self.db: PostgreSQLDB | None = None

    # def __post_init__(self):
    #     self.graph_name = self.namespace or os.environ.get("AGE_GRAPH_NAME", "lightrag")
    #     self._node_embed_algorithms = {
    #         "node2vec": self._node2vec_embed,
    #     }
    #     self.db: PostgreSQLDB | None = None

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    @staticmethod
    def _record_to_dict(record: asyncpg.Record) -> dict[str, Any]:
        """
        Convert a record returned from an age query to a dictionary

        Args:
            record (): a record from an age query result

        Returns:
            dict[str, Any]: a dictionary representation of the record where
                the dictionary key is the field name and the value is the
                value converted to a python type
        """
        # result holder
        d = {}

        # prebuild a mapping of vertex_id to vertex mappings to be used
        # later to build edges
        vertices = {}
        for k in record.keys():
            v = record[k]
            # agtype comes back '{key: value}::type' which must be parsed
            if isinstance(v, str) and "::" in v:
                if v.startswith("[") and v.endswith("]"):
                    if "::vertex" not in v:
                        continue
                    v = v.replace("::vertex", "")
                    vertexes = json.loads(v)
                    for vertex in vertexes:
                        vertices[vertex["id"]] = vertex.get("properties")
                else:
                    dtype = v.split("::")[-1]
                    v = v.split("::")[0]
                    if dtype == "vertex":
                        vertex = json.loads(v)
                        vertices[vertex["id"]] = vertex.get("properties")

        # iterate returned fields and parse appropriately
        for k in record.keys():
            v = record[k]
            if isinstance(v, str) and "::" in v:
                if v.startswith("[") and v.endswith("]"):
                    if "::vertex" in v:
                        v = v.replace("::vertex", "")
                        d[k] = json.loads(v)

                    elif "::edge" in v:
                        v = v.replace("::edge", "")
                        d[k] = json.loads(v)
                    else:
                        print("WARNING: unsupported type")
                        continue

                else:
                    dtype = v.split("::")[-1]
                    v = v.split("::")[0]
                    if dtype == "vertex":
                        d[k] = json.loads(v)
                    elif dtype == "edge":
                        d[k] = json.loads(v)
            else:
                try:
                    d[k] = (
                        json.loads(v)
                        if isinstance(v, str)
                        and (v.startswith("{") or v.startswith("["))
                        else v
                    )
                except json.JSONDecodeError:
                    d[k] = v

        return d

    @staticmethod
    def _format_properties(
        properties: dict[str, Any], _id: Union[str, None] = None
    ) -> str:
        """
        Convert a dictionary of properties to a string representation that
        can be used in a cypher query insert/merge statement.

        Args:
            properties (dict[str,str]): a dictionary containing node/edge properties
            _id (Union[str, None]): the id of the node or None if none exists

        Returns:
            str: the properties dictionary as a properly formatted string
        """
        props = []
        # wrap property key in backticks to escape
        for k, v in properties.items():
            prop = f"`{k}`: {json.dumps(v)}"
            props.append(prop)
        if _id is not None and "id" not in properties:
            props.append(
                f"id: {json.dumps(_id)}" if isinstance(_id, str) else f"id: {_id}"
            )
        return "{" + ", ".join(props) + "}"

    async def _query(
        self,
        query: str,
        readonly: bool = True,
        upsert: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Query the graph by taking a cypher query, converting it to an
        age compatible query, executing it and converting the result
        """
        # Check if db exists
        if self.db is None:
            logger.error("Database connection is None")
            await self.initialize()
            if self.db is None:
                raise PGGraphQueryException({"message": "Database connection is None", "detail": "Failed to initialize database"})
        
        try:
            if readonly:
                data = await self.db.query(
                    query,
                    multirows=True,
                    with_age=True,
                    graph_name=self.graph_name,
                )
            else:
                data = await self.db.execute(
                    query,
                    upsert=upsert,
                    with_age=True,
                    graph_name=self.graph_name,
                )

            if data is None:
                return []
            else:
                return [self._record_to_dict(d) for d in data]
                    
        except Exception as e:
            logger.error(f"Error in graph query: {str(e)}")
            raise PGGraphQueryException({
                "message": f"Error executing graph query: {query[:100]}...",
                "wrapped": query,
                "detail": str(e),
            }) from e

    async def has_node(self, node_id: str) -> bool:

        entity_name_label = node_id.strip('"')

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:base {entity_id: "%s"})
                     RETURN count(n) > 0 AS node_exists
                   $$) AS (node_exists bool)""" % (self.graph_name, entity_name_label)

        single_result = (await self._query(query))[0]

        return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        src_label = source_node_id.strip('"')
        tgt_label = target_node_id.strip('"')

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (a:base {entity_id: "%s"})-[r]-(b:base {entity_id: "%s"})
                     RETURN COUNT(r) > 0 AS edge_exists
                   $$) AS (edge_exists bool)""" % (
            self.graph_name,
            src_label,
            tgt_label,
        )

        single_result = (await self._query(query))[0]

        return single_result["edge_exists"]

    @performance
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Get multiple nodes in a single query"""
        if not node_ids:
            return {}
        
        # Track performance
        start_time = time.time()
        
        # Verify database connection
        if self.db is None or getattr(self.db, 'pool', None) is None or self.db.pool._closed:
            logger.error("Database connection is not available or closed")
            await self.initialize()  # Try to reinitialize
            if self.db is None or getattr(self.db, 'pool', None) is None or self.db.pool._closed:
                logger.error("Failed to reinitialize database connection")
                return {}
        
        # Create batch query
        encoded_ids = [self._encode_graph_label(node_id.strip('"')) for node_id in node_ids]
        id_list = ", ".join([f'"{node_id}"' for node_id in encoded_ids])
        
        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:Entity)
                    WHERE n.node_id IN [{id_list}]
                    RETURN n.node_id AS node_id, n AS node
                $$) AS (node_id text, node agtype)"""
        
        try:
            results = await self._query(query)
            
            # Build a map of original ids to node data
            nodes_map = {}
            for record in results:
                if "node_id" in record and "node" in record:
                    original_id = self._decode_graph_label(record["node_id"])
                    nodes_map[original_id] = record["node"]
            
            # Log performance for large batches
            if len(node_ids) > 10:
                logger.debug(
                    f"Batch retrieved {len(nodes_map)}/{len(node_ids)} nodes in {time.time() - start_time:.4f}s"
                )
                
            return nodes_map
        except Exception as e:
            logger.error(f"Error in batch node retrieval: {e}")
            return {}
    
    @performance
    async def get_edges_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
        """Get multiple edges in a single query"""
        if not edge_pairs:
            return {}
        
        # Track performance
        start_time = time.time()
        
        # Build query for multiple edges
        edge_conditions = []
        for src, tgt in edge_pairs:
            src_encoded = self._encode_graph_label(src.strip('"'))
            tgt_encoded = self._encode_graph_label(tgt.strip('"'))
            edge_conditions.append(f'(a.node_id = "{src_encoded}" AND b.node_id = "{tgt_encoded}")')
        
        conditions = " OR ".join(edge_conditions)
        
        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE {conditions}
                    RETURN a.node_id AS src_id, b.node_id AS tgt_id, properties(r) AS edge_data
                $$) AS (src_id text, tgt_id text, edge_data agtype)"""
        
        try:
            results = await self._query(query)
            
            # Build a map of edge pairs to edge data
            edges_map = {}
            for record in results:
                if "src_id" in record and "tgt_id" in record and "edge_data" in record:
                    src_id = self._decode_graph_label(record["src_id"])
                    tgt_id = self._decode_graph_label(record["tgt_id"])
                    edges_map[(src_id, tgt_id)] = record["edge_data"]
            
            # Log performance for large batches
            if len(edge_pairs) > 10:
                logger.debug(
                    f"Batch retrieved {len(edges_map)}/{len(edge_pairs)} edges in {time.time() - start_time:.4f}s"
                )
                
            return edges_map
        except Exception as e:
            logger.error(f"Error in batch edge retrieval: {e}")
            return {}

    @performance
    async def get_node_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Get edges for multiple nodes in a single query"""
        if not node_ids:
            return {}
        
        # Track performance
        start_time = time.time()
        
        # Build query for multiple nodes' edges
        encoded_ids = [self._encode_graph_label(node_id.strip('"')) for node_id in node_ids]
        id_list = ", ".join([f'"{node_id}"' for node_id in encoded_ids])
        
        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:Entity)-[r]-(m:Entity)
                    WHERE n.node_id IN [{id_list}]
                    RETURN n.node_id AS node_id, collect([m.node_id, type(r)]) AS connections
                $$) AS (node_id text, connections agtype)"""
        
        try:
            results = await self._query(query)
            
            # Build a map of node ids to edge lists
            edges_map = {node_id: [] for node_id in node_ids}
            for record in results:
                if "node_id" in record and "connections" in record:
                    src_id = self._decode_graph_label(record["node_id"])
                    connections = record["connections"]
                    edges = []
                    for conn in connections:
                        if len(conn) >= 2:
                            tgt_id = self._decode_graph_label(conn[0])
                            edges.append((src_id, tgt_id))
                    edges_map[src_id] = edges
            
            # Log performance for large batches
            if len(node_ids) > 10:
                edges_count = sum(len(edges) for edges in edges_map.values())
                logger.debug(
                    f"Batch retrieved {edges_count} edges for {len(edges_map)} nodes in {time.time() - start_time:.4f}s"
                )
                
            return edges_map
        except Exception as e:
            logger.error(f"Error in batch node edges retrieval: {e}")
            return {node_id: [] for node_id in node_ids}
    
    async def get_edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """Get degrees for multiple edges in a single query.
        
        This calculates the combined degree (number of connections) for the source and target 
        nodes of each edge. Higher degrees indicate more connected/central nodes in the graph.
        
        Args:
            edge_pairs: List of tuples containing (source_node_id, target_node_id)
            
        Returns:
            Dictionary mapping edge pairs to their combined degree
        """
        if not edge_pairs:
            return {}
        
        # Track performance
        start_time = time.time()
        
        # Build query components for all edges
        node_pairs_map = {}
        for src, tgt in edge_pairs:
            src_encoded = self._encode_graph_label(src.strip('"'))
            tgt_encoded = self._encode_graph_label(tgt.strip('"'))
            node_pairs_map[(src, tgt)] = (src_encoded, tgt_encoded)
        
        # Get all unique nodes to query
        all_nodes = set()
        for src, tgt in edge_pairs:
            all_nodes.add(src)
            all_nodes.add(tgt)
        
        node_ids = list(all_nodes)
        encoded_nodes = [self._encode_graph_label(node_id.strip('"')) for node_id in node_ids]
        node_list = ", ".join([f'"{node_id}"' for node_id in encoded_nodes])
        
        # Query for degrees of all nodes in a single operation
        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:Entity)
                    WHERE n.node_id IN [{node_list}]
                    OPTIONAL MATCH (n)-[r]->()
                    RETURN n.node_id AS node_id, count(r) AS degree
                $$) AS (node_id text, degree integer)"""
        
        try:
            results = await self._query(query)
            
            # Build degree map for each node
            node_degree_map = {}
            for result in results:
                if "node_id" in result and "degree" in result:
                    original_id = self._decode_graph_label(result["node_id"])
                    node_degree_map[original_id] = result["degree"]
            
            # Calculate combined degrees for each edge pair
            edge_degrees = {}
            for edge in edge_pairs:
                src, tgt = edge
                src_degree = node_degree_map.get(src, 0)
                tgt_degree = node_degree_map.get(tgt, 0)
                edge_degrees[edge] = src_degree + tgt_degree
            
            # Log performance for large batches
            if len(edge_pairs) > 10:
                logger.debug(
                    f"Batch retrieved degrees for {len(edge_degrees)} edges in {time.time() - start_time:.4f}s"
                )
            
            return edge_degrees
            
        except Exception as e:
            logger.error(f"Error in batch edge degree retrieval: {e}")
            return {edge: 0 for edge in edge_pairs}
    
    @performance
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node with caching"""
        cache_key = node_id
        if cache_key in self._node_cache:
            self._cache_hits += 1
            return self._node_cache[cache_key]
            
        self._cache_misses += 1
        result = await self._get_node_impl(node_id)
        
        # Manage cache size
        if len(self._node_cache) >= self._max_cache_size:
            # Remove oldest 10% of entries
            remove_count = max(1, self._max_cache_size // 10)
            for _ in range(remove_count):
                if self._node_cache:
                    self._node_cache.pop(next(iter(self._node_cache)), None)
                
        self._node_cache[cache_key] = result
        return result

    async def node_degree(self, node_id: str) -> int:
        label = node_id.strip('"')

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:base {entity_id: "%s"})-[]-(x)
                     RETURN count(x) AS total_edge_count
                   $$) AS (total_edge_count integer)""" % (self.graph_name, label)
        record = (await self._query(query))[0]
        if record:
            edge_count = int(record["total_edge_count"])
            return edge_count

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)

        return degrees

    @performance
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes"""

        src_label = source_node_id.strip('"')
        tgt_label = target_node_id.strip('"')

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (a:base {entity_id: "%s"})-[r]->(b:base {entity_id: "%s"})
                     RETURN properties(r) as edge_properties
                     LIMIT 1
                   $$) AS (edge_properties agtype)""" % (
            self.graph_name,
            src_label,
            tgt_label,
        )
        record = await self._query(query)
        if record and record[0] and record[0]["edge_properties"]:
            result = record[0]["edge_properties"]

            return result

    @performance
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]]:
      """
      Retrieves all edges (relationships) for a particular node identified by its label.
      :return: list of tuples containing (source, target) node IDs
      """
      print(Fore.BLUE + f"get_node_edges: {source_node_id}")
      try:
          label = self._encode_graph_label(source_node_id.strip('"'))
          
          # Query for outgoing edges
          outgoing_query = """SELECT * FROM cypher('%s', $$
                          MATCH (n:Entity {node_id: "%s"})-[]->(connected:Entity)
                          RETURN n, connected
                          LIMIT 100
                        $$) AS (n agtype, connected agtype)""" % (
              self.graph_name,
              label,
          )
          outgoing_results = await self._query(outgoing_query)
          
          # Query for incoming edges
          incoming_query = """SELECT * FROM cypher('%s', $$
                          MATCH (connected:Entity)-[]->(n:Entity {node_id: "%s"})
                          RETURN n, connected
                          LIMIT 100
                        $$) AS (n agtype, connected agtype)""" % (
              self.graph_name,
              label,
          )
          incoming_results = await self._query(incoming_query)

          # Combine results
          results = (outgoing_results or []) + (incoming_results or [])
          if not results:
              return []

          # Extract unique edges
          edges = []
          for record in results:
              source_node = record.get("n")
              connected_node = record.get("connected")

              if not source_node or not connected_node:
                  continue
                  
              source_label = source_node.get("node_id")
              target_label = connected_node.get("node_id")

              if source_label and target_label:
                  edge = (
                      self._decode_graph_label(source_label),
                      self._decode_graph_label(target_label),
                  )
                  if edge not in edges:
                      edges.append(edge)

          return edges
      except Exception as e:
          logger.error(f"Error getting edges for node {source_node_id}: {e}")
          return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the Neo4j database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        if "entity_id" not in node_data:
            raise ValueError(
                "PostgreSQL: node properties must contain an 'entity_id' field"
            )

        label = node_id.strip('"')
        properties = self._format_properties(node_data)

        query = """SELECT * FROM cypher('%s', $$
                     MERGE (n:base {entity_id: "%s"})
                     SET n += %s
                     RETURN n
                   $$) AS (n agtype)""" % (
            self.graph_name,
            label,
            properties,
        )

        try:
            await self._query(query, readonly=False, upsert=True)

        except Exception:
            logger.error(f"POSTGRES, upsert_node error on node_id: `{node_id}`")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): dictionary of properties to set on the edge
        """
        src_label = source_node_id.strip('"')
        tgt_label = target_node_id.strip('"')
        edge_properties = self._format_properties(edge_data)

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (source:base {entity_id: "%s"})
                     WITH source
                     MATCH (target:base {entity_id: "%s"})
                     MERGE (source)-[r:DIRECTED]->(target)
                     SET r += %s
                     RETURN r
                   $$) AS (r agtype)""" % (
            self.graph_name,
            src_label,
            tgt_label,
            edge_properties,
        )

        try:
            await self._query(query, readonly=False, upsert=True)

        except Exception:
            logger.error(
                f"POSTGRES, upsert_edge error on edge: `{source_node_id}`-`{target_node_id}`"
            )
            raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")

    async def delete_node(self, node_id: str) -> None:
        """
        Delete a node from the graph.

        Args:
            node_id (str): The ID of the node to delete.
        """
        label = node_id.strip('"')

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:base {entity_id: "%s"})
                     DETACH DELETE n
                   $$) AS (n agtype)""" % (self.graph_name, label)

        try:
            await self._query(query, readonly=False, upsert=False)
        except Exception as e:
            logger.error("Error during node deletion: {%s}", e)
            raise

    async def remove_nodes(self, node_ids: list[str]) -> None:
        """
        Remove multiple nodes from the graph.

        Args:
            node_ids (list[str]): A list of node IDs to remove.
        """
        node_ids = [node_id.strip('"') for node_id in node_ids]
        node_id_list = ", ".join([f'"{node_id}"' for node_id in node_ids])

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:base)
                     WHERE n.entity_id IN [%s]
                     DETACH DELETE n
                   $$) AS (n agtype)""" % (self.graph_name, node_id_list)

        try:
            await self._query(query, readonly=False, upsert=False)
        except Exception as e:
            logger.error("Error during node removal: {%s}", e)
            raise

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """
        Remove multiple edges from the graph.

        Args:
            edges (list[tuple[str, str]]): A list of edges to remove, where each edge is a tuple of (source_node_id, target_node_id).
        """
        for source, target in edges:
            src_label = source.strip('"')
            tgt_label = target.strip('"')

            query = """SELECT * FROM cypher('%s', $$
                         MATCH (a:base {entity_id: "%s"})-[r]->(b:base {entity_id: "%s"})
                         DELETE r
                       $$) AS (r agtype)""" % (self.graph_name, src_label, tgt_label)

            try:
                await self._query(query, readonly=False)
                logger.debug(f"Deleted edge from '{source}' to '{target}'")
            except Exception as e:
                logger.error(f"Error during edge deletion: {str(e)}")
                raise

    async def get_all_labels(self) -> list[str]:
        """
        Get all labels (node IDs) in the graph.

        Returns:
            list[str]: A list of all labels in the graph.
        """
        query = (
            """SELECT * FROM cypher('%s', $$
                     MATCH (n:base)
                     WHERE n.entity_id IS NOT NULL
                     RETURN DISTINCT n.entity_id AS label
                     ORDER BY n.entity_id
                   $$) AS (label text)"""
            % self.graph_name
        )

        results = await self._query(query)
        labels = [result["label"] for result in results]
        return labels

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        """
        Generate node embeddings using the specified algorithm.

        Args:
            algorithm (str): The name of the embedding algorithm to use.

        Returns:
            tuple[np.ndarray[Any, Any], list[str]]: A tuple containing the embeddings and the corresponding node IDs.
        """
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Unsupported embedding algorithm: {algorithm}")

        embed_func = self._node_embed_algorithms[algorithm]
        return await embed_func()

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node, * means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return, Defaults to 1000 (not BFS nor DFS garanteed)

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        # First, count the total number of nodes that would be returned without limit
        if node_label == "*":
            count_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:base)
                    RETURN count(distinct n) AS total_nodes
                    $$) AS (total_nodes bigint)"""
        else:
            strip_label = node_label.strip('"')
            count_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:base {{entity_id: "{strip_label}"}})
                    OPTIONAL MATCH p = (n)-[*..{max_depth}]-(m)
                    RETURN count(distinct m) AS total_nodes
                    $$) AS (total_nodes bigint)"""

        count_result = await self._query(count_query)
        total_nodes = count_result[0]["total_nodes"] if count_result else 0
        is_truncated = total_nodes > max_nodes

        # Now get the actual data with limit
        if node_label == "*":
            query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:base)
                    OPTIONAL MATCH (n)-[r]->(target:base)
                    RETURN collect(distinct n) AS n, collect(distinct r) AS r
                    LIMIT {max_nodes}
                    $$) AS (n agtype, r agtype)"""
        else:
            strip_label = node_label.strip('"')
            query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:base {{entity_id: "{strip_label}"}})
                    OPTIONAL MATCH p = (n)-[*..{max_depth}]-(m)
                    RETURN nodes(p) AS n, relationships(p) AS r
                    LIMIT {max_nodes}
                    $$) AS (n agtype, r agtype)"""

        results = await self._query(query)

        # Process the query results with deduplication by node and edge IDs
        nodes_dict = {}
        edges_dict = {}
        for result in results:
            # Handle single node cases
            if result.get("n") and isinstance(result["n"], dict):
                node_id = str(result["n"]["id"])
                if node_id not in nodes_dict:
                    nodes_dict[node_id] = KnowledgeGraphNode(
                        id=node_id,
                        labels=[result["n"]["properties"]["entity_id"]],
                        properties=result["n"]["properties"],
                    )
            # Handle node list cases
            elif result.get("n") and isinstance(result["n"], list):
                for node in result["n"]:
                    if isinstance(node, dict) and "id" in node:
                        node_id = str(node["id"])
                        if node_id not in nodes_dict and "properties" in node:
                            nodes_dict[node_id] = KnowledgeGraphNode(
                                id=node_id,
                                labels=[node["properties"]["entity_id"]],
                                properties=node["properties"],
                            )

            # Handle single edge cases
            if result.get("r") and isinstance(result["r"], dict):
                edge_id = str(result["r"]["id"])
                if edge_id not in edges_dict:
                    edges_dict[edge_id] = KnowledgeGraphEdge(
                        id=edge_id,
                        type="DIRECTED",
                        source=str(result["r"]["start_id"]),
                        target=str(result["r"]["end_id"]),
                        properties=result["r"]["properties"],
                    )
            # Handle edge list cases
            elif result.get("r") and isinstance(result["r"], list):
                for edge in result["r"]:
                    if isinstance(edge, dict) and "id" in edge:
                        edge_id = str(edge["id"])
                        if edge_id not in edges_dict:
                            edges_dict[edge_id] = KnowledgeGraphEdge(
                                id=edge_id,
                                type="DIRECTED",
                                source=str(edge["start_id"]),
                                target=str(edge["end_id"]),
                                properties=edge["properties"],
                            )

        # Construct and return the KnowledgeGraph with deduplicated nodes and edges
        kg = KnowledgeGraph(
            nodes=list(nodes_dict.values()),
            edges=list(edges_dict.values()),
            is_truncated=is_truncated,
        )

        logger.info(
            f"Subgraph query successful | Node count: {len(kg.nodes)} | Edge count: {len(kg.edges)}"
        )
        return kg

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            drop_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                              MATCH (n)
                              DETACH DELETE n
                            $$) AS (result agtype)"""

            await self._query(drop_query, readonly=False)
            return {"status": "success", "message": "graph data dropped"}
        except Exception as e:
            logger.error(f"Error dropping graph: {e}")
            return {"status": "error", "message": str(e)}


NAMESPACE_TABLE_MAP = {
    NameSpace.KV_STORE_FULL_DOCS: "LIGHTRAG_DOC_FULL",
    NameSpace.KV_STORE_TEXT_CHUNKS: "LIGHTRAG_DOC_CHUNKS",
    NameSpace.VECTOR_STORE_CHUNKS: "LIGHTRAG_DOC_CHUNKS",
    NameSpace.VECTOR_STORE_ENTITIES: "LIGHTRAG_VDB_ENTITY",
    NameSpace.VECTOR_STORE_RELATIONSHIPS: "LIGHTRAG_VDB_RELATION",
    NameSpace.DOC_STATUS: "LIGHTRAG_DOC_STATUS",
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE: "LIGHTRAG_LLM_CACHE",
}


def namespace_to_table_name(namespace: str) -> str:
    for k, v in NAMESPACE_TABLE_MAP.items():
        if is_namespace(namespace, k):
            return v


TABLES = {
    "LIGHTRAG_DOC_FULL": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_FULL (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    doc_name VARCHAR(1024),
                    content TEXT,
                    meta JSONB,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP,
	                CONSTRAINT LIGHTRAG_DOC_FULL_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_DOC_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    content_vector VECTOR,
                    file_path VARCHAR(256),
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP,
	                CONSTRAINT LIGHTRAG_DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
                    );
                    
                    -- Create index for full_doc_id lookups
                    CREATE INDEX IF NOT EXISTS idx_doc_chunks_full_doc_id 
                    ON LIGHTRAG_DOC_CHUNKS(workspace, full_doc_id);
                    
                    -- Create optimized vector index for similarity search
                    CREATE INDEX IF NOT EXISTS idx_chunks_vector 
                    ON LIGHTRAG_DOC_CHUNKS USING ivfflat (content_vector vector_cosine_ops) 
                    WITH (lists = 100);
                    """
    },
    "LIGHTRAG_VDB_ENTITY": {
        "ddl": """CREATE TABLE LIGHTRAG_VDB_ENTITY (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(255),
                    content TEXT,
                    content_vector VECTOR,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
	                CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
                    );
                    
                    -- Add index for entity lookups
                    CREATE INDEX IF NOT EXISTS idx_entity_entity_name 
                    ON LIGHTRAG_VDB_ENTITY(workspace, entity_name);
                    
                    -- Create optimized vector index for similarity search
                    CREATE INDEX IF NOT EXISTS idx_entity_vector 
                    ON LIGHTRAG_VDB_ENTITY USING ivfflat (content_vector vector_cosine_ops) 
                    WITH (lists = 100);
                    """
    },
    "LIGHTRAG_VDB_RELATION": {
        "ddl": """CREATE TABLE LIGHTRAG_VDB_RELATION (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(256),
                    target_id VARCHAR(256),
                    content TEXT,
                    content_vector VECTOR,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
	                CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
                    );
                    
                    -- Add indexes for relationship lookups
                    CREATE INDEX IF NOT EXISTS idx_relation_source_id 
                    ON LIGHTRAG_VDB_RELATION(workspace, source_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_relation_target_id 
                    ON LIGHTRAG_VDB_RELATION(workspace, target_id);
                    
                    -- Create optimized vector index for similarity search
                    CREATE INDEX IF NOT EXISTS idx_relation_vector 
                    ON LIGHTRAG_VDB_RELATION USING ivfflat (content_vector vector_cosine_ops) 
                    WITH (lists = 100);
                    """
    },
    "LIGHTRAG_LLM_CACHE": {
        "ddl": """CREATE TABLE LIGHTRAG_LLM_CACHE (
	                workspace varchar(255) NOT NULL,
	                id varchar(255) NOT NULL,
	                mode varchar(32) NOT NULL,
                    original_prompt TEXT,
                    return_value TEXT,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP,
	                CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (workspace, mode, id)
                    );
                    
                    -- Add index for cache lookups by mode
                    CREATE INDEX IF NOT EXISTS idx_llm_cache_mode 
                    ON LIGHTRAG_LLM_CACHE(workspace, mode);
                    """
    },
    "LIGHTRAG_DOC_STATUS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_STATUS (
	               workspace varchar(255) NOT NULL,
	               id varchar(255) NOT NULL,
	               content TEXT NULL,
	               content_summary varchar(255) NULL,
	               content_length int4 NULL,
	               chunks_count int4 NULL,
	               status varchar(64) NULL,
	               file_path TEXT NULL,
	               created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	               updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	               CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
	              );
	              
	              -- Add index for status lookups
                  CREATE INDEX IF NOT EXISTS idx_doc_status_status 
                  ON LIGHTRAG_DOC_STATUS(workspace, status);
                  """
    },
}


SQL_TEMPLATES = {
    # SQL for KVStorage - using explicit varchar(255) typing for better index utilization
    "get_by_id_full_docs": """SELECT id, COALESCE(content, '') as content
                                FROM LIGHTRAG_DOC_FULL WHERE workspace=$1::varchar(255) AND id=$2::varchar(255)
                            """,
    "get_by_id_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                chunk_order_index, full_doc_id
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1::varchar(255) AND id=$2::varchar(255)
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1::varchar(255) AND mode=$2::varchar(32)
                               """,
    "get_by_mode_id_llm_response_cache": """SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
                           FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1::varchar(255) AND mode=$2::varchar(32) AND id=$3::varchar(255)
                          """,
    "get_by_ids_full_docs": """SELECT id, COALESCE(content, '') as content
                              FROM LIGHTRAG_DOC_FULL
                              WHERE workspace=$1::varchar(255) AND id = ANY($2::varchar(255)[])
                            """,
    "get_by_ids_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content, chunk_order_index, full_doc_id
                                FROM LIGHTRAG_DOC_CHUNKS
                                WHERE workspace=$1::varchar(255) AND id = ANY($2::varchar(255)[])
                              """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
                                      FROM LIGHTRAG_LLM_CACHE
                                      WHERE workspace=$1::varchar(255) AND id = ANY($2::varchar(255)[])
                                    """,
    "vector_query_chunks": """
            WITH vector_search AS (
                SELECT c.id, c.content, c.full_doc_id, c.file_path, c.content_vector <=> $4::vector AS distance
                FROM LIGHTRAG_DOC_CHUNKS c
                WHERE c.workspace=$1::varchar(255)
                AND ($3::varchar(255)[] IS NULL OR c.full_doc_id = ANY($3::varchar(255)[]))
                ORDER BY distance
                LIMIT $2::int
            )
            SELECT id, content, full_doc_id, file_path, distance
            FROM vector_search
            """,
    "vector_query_entities": """
            WITH vector_search AS (
                SELECT e.id, e.entity_name, e.content, e.chunk_ids, e.file_path,
                    e.content_vector <=> $4::vector AS distance
                FROM LIGHTRAG_VDB_ENTITY e
                WHERE e.workspace=$1::varchar(255)
                ORDER BY distance
                LIMIT $2::int * 2  -- Fetch more candidates for filtering
            )
            SELECT vs.id, vs.entity_name, vs.content, vs.file_path, vs.distance
            FROM vector_search vs
            WHERE $3::varchar(255)[] IS NULL OR
                EXISTS (
                    SELECT 1 FROM LIGHTRAG_DOC_CHUNKS c
                    WHERE c.workspace=$1::varchar(255)
                    AND c.id = ANY(vs.chunk_ids)
                    AND c.full_doc_id = ANY($3::varchar(255)[])
                )
            LIMIT $2::int
            """,
    "vector_query_relationships": """
            WITH vector_search AS (
                SELECT r.id, r.source_id as src_id, r.target_id as tgt_id, 
                    r.content, r.chunk_ids, r.file_path,
                    r.content_vector <=> $4::vector AS distance
                FROM LIGHTRAG_VDB_RELATION r
                WHERE r.workspace=$1::varchar(255)
                ORDER BY distance
                LIMIT $2::int * 2  -- Fetch more candidates for filtering
            )
            SELECT vs.id, vs.src_id, vs.tgt_id, vs.content, vs.file_path, vs.distance
            FROM vector_search vs
            WHERE $3::varchar(255)[] IS NULL OR
                EXISTS (
                    SELECT 1 FROM LIGHTRAG_DOC_CHUNKS c
                    WHERE c.workspace=$1::varchar(255)
                    AND c.id = ANY(vs.chunk_ids)
                    AND c.full_doc_id = ANY($3::varchar(255)[])
                )
            LIMIT $2::int
            """,
    "upsert_doc_full": """INSERT INTO LIGHTRAG_DOC_FULL (id, content, workspace)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (workspace,id) DO UPDATE
                           SET content = $2, update_time = CURRENT_TIMESTAMP
                       """,
    "upsert_llm_response_cache": """INSERT INTO LIGHTRAG_LLM_CACHE(workspace,id,original_prompt,return_value,mode)
                                      VALUES ($1, $2, $3, $4, $5)
                                      ON CONFLICT (workspace,mode,id) DO UPDATE
                                      SET original_prompt = EXCLUDED.original_prompt,
                                      return_value=EXCLUDED.return_value,
                                      mode=EXCLUDED.mode,
                                      update_time = CURRENT_TIMESTAMP
                                     """,
    "upsert_chunk": """INSERT INTO LIGHTRAG_DOC_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector, file_path)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      file_path=EXCLUDED.file_path,
                      update_time = CURRENT_TIMESTAMP
                     """,
    # SQL for VectorStorage
    "upsert_entity": """INSERT INTO LIGHTRAG_VDB_ENTITY (workspace, id, entity_name, content,
                      content_vector, chunk_ids, file_path)
                      VALUES ($1, $2, $3, $4, $5, $6::varchar[], $7)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_name=EXCLUDED.entity_name,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      chunk_ids=EXCLUDED.chunk_ids,
                      file_path=EXCLUDED.file_path,
                      update_time=CURRENT_TIMESTAMP
                     """,
    "upsert_relationship": """INSERT INTO LIGHTRAG_VDB_RELATION (workspace, id, source_id,
                      target_id, content, content_vector, chunk_ids, file_path)
                      VALUES ($1, $2, $3, $4, $5, $6, $7::varchar[], $8)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET source_id=EXCLUDED.source_id,
                      target_id=EXCLUDED.target_id,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      chunk_ids=EXCLUDED.chunk_ids,
                      file_path=EXCLUDED.file_path,
                      update_time = CURRENT_TIMESTAMP
                     """,
    "relationships": """
    WITH relevant_chunks AS (
        SELECT id as chunk_id
        FROM LIGHTRAG_DOC_CHUNKS
        WHERE {doc_ids} IS NULL OR full_doc_id = ANY(ARRAY[{doc_ids}])
    )
    SELECT source_id as src_id, target_id as tgt_id
    FROM (
        SELECT r.id, r.source_id, r.target_id, 1 - (r.content_vector <=> '[{embedding_string}]'::vector) as distance
        FROM LIGHTRAG_VDB_RELATION r
        JOIN relevant_chunks c ON c.chunk_id = ANY(r.chunk_ids)
        WHERE r.workspace=$1
    ) filtered
    WHERE distance>$2
    ORDER BY distance DESC
    LIMIT $3
    """,
    "entities": """
        WITH relevant_chunks AS (
            SELECT id as chunk_id
            FROM LIGHTRAG_DOC_CHUNKS
            WHERE {doc_ids} IS NULL OR full_doc_id = ANY(ARRAY[{doc_ids}])
        )
        SELECT entity_name FROM
            (
                SELECT e.id, e.entity_name, 1 - (e.content_vector <=> '[{embedding_string}]'::vector) as distance
                FROM LIGHTRAG_VDB_ENTITY e
                JOIN relevant_chunks c ON c.chunk_id = ANY(e.chunk_ids)
                WHERE e.workspace=$1
            )
        WHERE distance>$2
        ORDER BY distance DESC
        LIMIT $3
    """,
    "chunks": """
        WITH relevant_chunks AS (
            SELECT id as chunk_id
            FROM LIGHTRAG_DOC_CHUNKS
            WHERE {doc_ids} IS NULL OR full_doc_id = ANY(ARRAY[{doc_ids}])
        )
        SELECT id, content, file_path FROM
            (
                SELECT id, content, file_path, 1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
                FROM LIGHTRAG_DOC_CHUNKS
                where workspace=$1
                AND id IN (SELECT chunk_id FROM relevant_chunks)
            ) as chunk_distances
            WHERE distance>$2
            ORDER BY distance DESC
            LIMIT $3
    """,
    # DROP tables
    "drop_specifiy_table_workspace": """
        DELETE FROM {table_name} WHERE workspace=$1
       """,
}
